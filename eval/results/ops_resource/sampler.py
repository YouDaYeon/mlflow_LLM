"""1초 간격 리소스 샘플러. usage: sampler.py OUT.jsonl  (SIGTERM까지)"""
import json, os, subprocess, sys, time, threading, signal
OUT = sys.argv[1]
HZ = os.sysconf("SC_CLK_TCK"); PAGE = os.sysconf("SC_PAGE_SIZE")
# 측정 시점의 llama-server PID. 다시 쓸 때는 ps로 확인해 바꾼다.
FIXED = {"llm_qwen35b": 3218703, "llm_gemma26b": 43208}
CONTAINERS = ["mlflow_llm_dy-backend-1","mlflow_llm_dy-frontend-1","mlflow_llm_dy-mlflow-1","mlflow_llm_dy-postgres-1","mlflow_llm_dy-pgadmin-1"]
stop = False
signal.signal(signal.SIGTERM, lambda *a: globals().__setitem__("stop", True))
def eval_pids():
    out = subprocess.run(["pgrep","-f","python.*measure_run.py --label"],capture_output=True,text=True).stdout.split()
    return [int(p) for p in out if int(p)!=os.getpid()]
def proc(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")",1)[1].split()
        return int(f[11])+int(f[12]), int(f[21])*PAGE
    except Exception: return None
docker_last = {}
def docker_loop():
    while not stop:
        r = subprocess.run(["docker","stats","--no-stream","--format","{{json .}}",*CONTAINERS],capture_output=True,text=True).stdout
        d = {}
        for line in r.splitlines():
            j = json.loads(line); d[j["Name"]] = {"cpu": j["CPUPerc"], "mem": j["MemUsage"], "net": j["NetIO"], "blk": j["BlockIO"]}
        docker_last.clear(); docker_last.update(d); docker_last["_t"] = time.time()
threading.Thread(target=docker_loop, daemon=True).start()
prev = {}
with open(OUT, "a") as fo:
    while not stop:
        t = time.time(); rec = {"t": t, "proc": {}}
        pids = dict(FIXED); 
        for i,p in enumerate(eval_pids()): pids[f"eval_{i}"] = p
        for name,pid in pids.items():
            s = proc(pid)
            if not s: continue
            cpu = None
            if pid in prev:
                pt, pc = prev[pid]; cpu = round((s[0]-pc)/HZ/(t-pt)*100,1)
            prev[pid] = (t, s[0]); rec["proc"][name] = {"cpu": cpu, "rss_mb": round(s[1]/2**20)}
        g = subprocess.run(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,power.draw","--format=csv,noheader,nounits","-i","5,7"],capture_output=True,text=True).stdout
        rec["gpu"] = [dict(zip(["idx","util","mem_mb","power_w"],[x.strip() for x in l.split(",")])) for l in g.splitlines()]
        rec["docker"] = dict(docker_last)
        fo.write(json.dumps(rec)+"\n"); fo.flush()
        time.sleep(max(0, 1-(time.time()-t)))
