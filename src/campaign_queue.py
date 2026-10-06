"""Coordinate run claims on a shared filesystem supporting locks across workers."""
import fcntl, json, os, socket, time
from pathlib import Path


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


class WorkQueue:
    def __init__(self, root):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.tasks=self.manifest['tasks']
        for name in ['locks','records','owners','failures','runs']:
            (self.root/name).mkdir(exist_ok=True)

    def failures(self, key):
        path=self.root/'failures'/f'{key}.json'
        return json.loads(path.read_text()) if path.exists() else []

    def claim(self):
        for task in self.tasks:
            key=task['id']
            if (self.root/'records'/f'{key}.json').exists() or len(self.failures(key))>=3: continue
            f=(self.root/'locks'/f'{key}.lock').open('a+')
            try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                f.close(); continue
            if (self.root/'records'/f'{key}.json').exists() or len(self.failures(key))>=3:
                f.close(); continue
            owner=dict(task=key, job_id=os.environ.get('SLURM_JOB_ID'),
                       array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                       array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                       qos=os.environ.get('SLURM_JOB_QOS'), host=socket.gethostname(),
                       pid=os.getpid(), claimed_at=time.time())
            atomic_json(self.root/'owners'/f'{key}.json',owner)
            return task,f,owner
        return None

    def complete(self, task, record):
        atomic_json(self.root/'records'/f"{task['id']}.json",record)

    def fail(self, task, error):
        path=self.root/'failures'/f"{task['id']}.json"
        previous=self.failures(task['id'])
        atomic_json(path,previous+[dict(time=time.time(),error=error,job=os.environ.get('SLURM_JOB_ID'))])

    def status(self):
        result=dict(total=len(self.tasks),completed=0,running=0,pending=0,failed=0)
        for task in self.tasks:
            key=task['id']
            if (self.root/'records'/f'{key}.json').exists(): result['completed']+=1; continue
            if len(self.failures(key))>=3: result['failed']+=1; continue
            with (self.root/'locks'/f'{key}.lock').open('a+') as f:
                try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: result['running']+=1
                else: result['pending']+=1
        return result
