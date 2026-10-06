"""Re-score saved MLP weights on the expanded held-out set, without training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from image_data import mlp_split_indices, load_extra_validation, MLP_SPLIT_VERSION
from train_vision import MLP, load_dataset


def evaluate(record, data_dir, device='cpu'):
    if record['arch'] != 'mlp':
        raise ValueError('Only MLP checkpoints are supported')
    x, y = load_dataset(record['task'], data_dir)
    train, valid = mlp_split_indices(len(x))
    extra_x, extra_y, extra_indices, extra_meta = load_extra_validation(record['task'], data_dir, len(x))
    with np.load(record['mechanism']['probe_context']) as context:
        np.testing.assert_array_equal(context['training_indices'], train)
        np.testing.assert_array_equal(context['validation_indices'][:len(valid)], valid)
        mu, sd = context['normalization_mean'], context['normalization_std']
        np.testing.assert_allclose(x[train].mean(0), mu, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(x[train].std(0)+1e-6, sd, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose((x[valid[:512]]-mu)/sd, context['target_probe'], rtol=1e-6, atol=1e-6)
    checkpoint = torch.load(record['checkpoint'], map_location='cpu', weights_only=False, mmap=False)
    for key in ['task', 'width', 'depth', 'ord', 'seed', 'config', 'lr', 'batch', 'phase1', 'phase2']:
        if checkpoint['meta'][key] != record[key]:
            raise ValueError(f'Checkpoint differs from record: {key}')
    if checkpoint['step'] != record['phase1'] + record['phase2']:
        raise ValueError('Checkpoint is not the final training step')
    net = MLP(x.shape[1], int(y.max()+1), record['width'], record['depth']).to(device)
    net.load_state_dict(checkpoint['model'], strict=True)
    net.eval()
    features = np.concatenate([x[valid], extra_x])
    labels = np.concatenate([y[valid], extra_y])
    with torch.no_grad():
        predictions = np.concatenate([net.good(torch.as_tensor((features[i:i+256]-mu)/sd, device=device)).argmax(-1).cpu().numpy()
                                      for i in range(0, len(features), 256)])
    original_accuracy = float(np.mean(predictions[:len(valid)] == y[valid]))
    # Stored accuracy was rounded from a GPU float32 mean. Half-way rounding
    # may differ by 0.0001; even one changed prediction (1/4000) still fails.
    if abs(original_accuracy - record['val_acc_full']) > 0.0000501:
        raise ValueError(f'Original accuracy did not reproduce: {original_accuracy} != {record["val_acc_full"]}')
    result = copy.deepcopy(record)
    result['previous_evaluation'] = dict(val_acc_full=record['val_acc_full'], validation_count=len(valid))
    result['vacc_validation_count'] = len(valid)  # Historical learning curve is NOT relabeled.
    result['val_acc_full'] = round(float(np.mean(predictions == labels)), 4)
    result['validation'] = dict(protocol=MLP_SPLIT_VERSION, training_count=len(train), validation_count=len(labels),
                                original_validation_count=len(valid), added_validation_count=len(extra_x), extra_source=extra_meta)
    result['reevaluation'] = dict(checkpoint_sha256=hashlib.sha256(Path(record['checkpoint']).read_bytes()).hexdigest(),
                                 correct=int(np.sum(predictions == labels)), count=len(labels),
                                 original_accuracy_unrounded=original_accuracy,
                                 additional_test_indices=extra_indices.tolist(), training_performed=False,
                                 overlap_probes='Unchanged saved 512-image probes; expanded set is for accuracy')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    args=parser.parse_args()
    result=evaluate(json.loads(args.record.read_text()), args.data_dir, args.device)
    result['source_record']=str(args.record.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
