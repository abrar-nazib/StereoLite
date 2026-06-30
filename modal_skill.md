---
name: modal-expert
description: Use when working with Modal.com cloud (running Python on remote GPUs/CPUs, managing Volumes, Secrets, scheduled jobs, web endpoints): triggered by `modal run`, `modal deploy`, `modal volume`, `modal app`, `modal shell`, `modal.com` URLs, `import modal`, references to A100/H100/H200/B200 cloud GPUs in a Modal context, `~/.modal.toml`, or asks like "run this on Modal", "train on A100", "spin up a Modal job". Skip for unrelated cloud providers (AWS Lambda, Vertex, RunPod, Lambda Labs, Replicate, Banana, etc.): those are not Modal.
---

# Modal Expert

Comprehensive reference for using Modal.com from the CLI and Python SDK. Source-of-truth synthesis from the official docs as of 2026-04-29. Treat this as a cheat sheet: not a tutorial.

## When this skill activates

User is using Modal to run Python remotely. Common asks:
- "run this script on Modal", "train on a Modal A100", "deploy this to Modal"
- managing Modal Volumes (download/upload data, persist checkpoints across runs)
- scheduling jobs (cron / period)
- web endpoints / serverless inference
- monitoring long detached training jobs
- diagnosing Modal-specific errors (gRPC payload, volume busy, container kills)

## 0. Mental model

- **Functions are remote.** `@app.function()` decorates a Python function; calling `f.remote()` runs it in a fresh container on Modal's infra. CPU/GPU/memory is allocated **per invocation**, container shuts down after `scaledown_window` (default 60 s).
- **Three execution modes:** `modal run` (ephemeral, dies with local process unless `--detach`), `modal deploy` (persistent app, gets stable name, supports cron/web/from_name), `modal serve` (hot-reload dev for web endpoints).
- **Volumes** are network filesystems that persist across runs. Mount with `volumes={"/mnt/data": vol}`. Writes need `vol.commit()` for synchronous visibility; readers need `vol.reload()` to see other containers' commits.
- **Images** are container images, defined declaratively in Python (chainable). Layer-cached; breaking layer N rebuilds N..end.
- **Secrets** are encrypted KV bags injected as env vars.
- **Cls** is a stateful Function: same container reused across method calls until idle, supports lifecycle hooks (`@enter`, `@exit`).
- **Sandbox** is for one-off shell commands (`nvidia-smi`, `python -c ...`).
- **Control plane is us-east-1** regardless of compute region: adds input/output latency.

## 1. Setup & auth

```bash
pip install modal
modal token new                     # browser OAuth, one-time per machine
modal token info                    # show active
modal profile list / activate NAME  # multiple workspaces
modal environment list / create NAME / delete NAME
```

Config at `~/.modal.toml`. Env: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MODAL_ENVIRONMENT`. Every CLI command takes `-e/--env`.

## 2. Daily workflow recipes

### Run a Python script on Modal once
```python
# script.py
import modal
app = modal.App("my-job")

@app.function()
def hello():
    return "ok"

@app.local_entrypoint()
def main():
    print(hello.remote())
```
```bash
modal run script.py
```

### Long training run, monitorable, laptop-can-sleep
```bash
modal run -d train.py::train --epochs 50      # -d = --detach
modal app list                                 # see running apps
modal app logs my-job --follow                 # tail (Ctrl+C detaches, training continues)
modal app logs my-job --since 1h --search "epe"   # filtered grep
modal app stop my-job -y                       # kill
```

### Volume: download data once, train many times
```python
vol = modal.Volume.from_name("training-data", create_if_missing=True)
img = modal.Image.debian_slim().apt_install("wget")

@app.function(image=img, volumes={"/data": vol}, timeout=12*3600)
def download(url: str):
    import subprocess
    subprocess.check_call(["wget", "-c", "-O", f"/data/{url.rsplit('/',1)[-1]}", url])
    vol.commit()                # CRITICAL: writes don't auto-publish synchronously
```

### Pull files from Modal back to local
```bash
modal volume ls training-data
modal volume get training-data /ckpt/best.pth ./best.pth
modal volume put training-data ./local.bin /uploaded.bin -f
modal volume rm training-data /old_ckpt -r
```

### GPU: only billed during the call
```python
@app.function(gpu="A100-40GB", timeout=4*3600, volumes={"/data": vol})
def train():
    import subprocess
    subprocess.run(["torchrun", "--nproc_per_node=1", "train.py"], check=True)
```
GPU spins up on call, shuts down when function returns + idle timeout (default 60 s). Zero idle cost.

### Multi-GPU DDP on one node
```python
@app.function(gpu="A100-80GB:4", timeout=24*3600, volumes={"/data": vol})
def train_ddp():
    subprocess.run(["torchrun", "--nproc_per_node=4", "train.py"], check=True)
```
Frameworks like PyTorch Lightning that re-exec the entrypoint need `ddp_spawn` strategy or subprocess launch. Multi-NODE training is private beta: contact `support@modal.com`.

### Persistent app with cron
```python
@app.function(schedule=modal.Cron("0 8 * * *"))   # 8 AM daily, stable across redeploys
def daily(): ...

@app.function(schedule=modal.Period(hours=6))     # every 6 h, RESETS on redeploy
def heartbeat(): ...
```
```bash
modal deploy app.py --name daily-job
```

### Sandbox: one-off command
```bash
modal shell --gpu a100 -c "nvidia-smi"
modal shell --image nvidia/cuda:12.4.1-devel-ubuntu22.04 --pty
```

### Cross-app function reference
```python
predict = modal.Function.from_name("predictor-app", "Predictor.infer")
predict.remote(payload)
```

## 3. CLI reference (one line each)

### `modal run`
Ephemeral run of a function or `@local_entrypoint`. Format: `file::func`.
- `-d, --detach`: survive local disconnect (load-bearing for long jobs)
- `-q, --quiet`: no progress bars
- `-i, --interactive`
- `-w, --write-result PATH`: save return value locally
- `-m`: interpret as Python module
- `--timestamps`: log timestamps
- `-e, --env TEXT`

### `modal deploy`
Persistent named deployment. Required for cron, web endpoints, cross-app refs.
- `--name TEXT`, `--tag TEXT`
- `--strategy [rolling|recreate]` (default `rolling`)
- `--stream-logs` (default off)

### `modal app`
- `list [--json]`: running/recent apps
- `logs APP [-f] [--since 2h] [-n 100] [--search T] [--function F] [--container C] [-s {stdout,stderr,system}] [--timestamps]`
- `stop APP [-y]`: kill all containers
- `history APP [--json]`: deployment versions
- `rollback APP [VERSION]`: to previous (or numbered) deploy
- `rollover APP --strategy {rolling|recreate}`: fresh containers, no code change
- `dashboard APP`: open browser

### `modal volume`
- `create NAME [--version 1|2]`
- `list`: all volumes; `ls VOL [PATH]`: files inside
- `get VOL REMOTE [LOCAL]`: `--force`; `-` = stdout
- `put VOL LOCAL [REMOTE] [-f]`
- `rm VOL PATH [-r]`
- `cp VOL SRC DST [-r]`: within volume
- `delete NAME [--allow-missing] [-y]`
- `rename OLD NEW [-y]`
- `dashboard VOL`

### `modal secret`
- `list`, `create NAME K=V K=V...`, `delete NAME`
- `--from-dotenv PATH`, `--from-json PATH`, `--force`

### `modal shell`
Drop into a container. Default cmd `/bin/bash`.
- `-c, --cmd TEXT`: run cmd instead of bash
- `--image TEXT`, `--add-python TEXT`
- `--volume NAME` (repeatable, mounts at `/mnt/{name}`)
- `--add-local PATH` (repeatable, `/mnt/{basename}`)
- `--secret NAME` (repeatable)
- `--cpu INT`, `--memory MIB`, `--gpu TEXT` (e.g. `a100:4`)
- `--cloud {aws,gcp,oci,auto}`, `--region TEXT`
- `--pty`, `-m`
- `modal shell file.py::fn`: drop into that function's exact env

### `modal container`
- `list [--app-id ID] [--json]`
- `logs CID [-f] [--all] [--since] [-n N] [-s SRC]`: default last 100
- `exec CID -- CMD ...`: note `--` before flagged sub-cmd. `--pty/--no-pty`.
- `stop CID [-y]`: SIGINT, reassigns inputs

### `modal token / profile / environment`
See §1.

### `modal dict` / `modal queue`
Persistent KV store / queue. CRUD-style: `create NAME`, `list`, `get/put/delete`, `clear NAME -y`. Queue adds `peek N`, `len [-t]`.

### `modal launch`, `modal nfs`
Not documented in current corpus: likely renamed/deprecated. `modal volume` replaces `nfs` (network-file-system).

## 4. Python SDK reference

### App
```python
app = modal.App("name", image=base, secrets=[...], volumes={...})
```
- `app.function(...)` / `app.cls(...)`: decorators
- `app.local_entrypoint()`: entry for `modal run`
- `app.run()`: context manager for ephemeral execution from a script
- `app.deploy()`: programmatic deploy
- `App.lookup(name, create_if_missing=False)`
- `app.include(other_app)`, `app.set_tags(...)`

### `@app.function()` kwargs (full list: most-used in **bold**)
- **`image`**: `modal.Image`
- **`gpu`**: string: `"A100"`, `"A100-40GB"`, `"A100-80GB"`, `"H100"`, `"H100!"` (no auto-upgrade), `"H200"`, `"B200"`, `"B200+"` (allows B300, billed B200), `"L4"`, `"L40S"`, `"T4"`, `"A10"`, `"RTX-PRO-6000"`. Multi-GPU: `:N` (max 8; A10 max 4; total VRAM cap 1,536 GB)
- **`cpu`**: physical cores, fractional ok (min 0.125)
- **`memory`**: MiB
- **`timeout`**: seconds. Default **300**, **range 10-86,400** (verified: raises `InvalidError` outside; 24 h is the hard ceiling). Each retry gets a fresh window.
- `startup_timeout`: separate init timeout (v1.1.4+)
- **`secrets=[modal.Secret.from_name("..."), ...]`**
- **`volumes={"/mnt/data": vol}`**: `read_only=True` supported
- `retries`: int or `modal.Retries(max_retries=N, backoff_coefficient=..., initial_delay=..., max_delay=...)`. **`initial_delay` must be 0-60 s (verified: raises `InvalidError` if >60).**
- `schedule`: `modal.Cron("0 8 * * *")` or `modal.Period(hours=1)`
- `min_containers`: keep N warm; `max_containers`: cap; `buffer_containers`: extra warm during bursts
- **`scaledown_window`**: idle-shutdown seconds (default **60**, range 2-1200). Older alias: `container_idle_timeout`.
- `enable_memory_snapshot=True`: fast cold start via snapshot of post-`@enter(snap=True)` state
- `experimental_options={"enable_gpu_snapshot": True}`: alpha
- `region`: `"us"|"eu"|"uk"|"ap"|"ca"|"me"|"sa"|"af"|"mx"` or list. **CA/SA/ME/MX/AF cost 2.5×, others 1.25×.**
- `cloud`: `"aws"|"gcp"|"oci"|"auto"`
- `name`: override
- `serialized=True`: pickle-define instead of import
- `mounts`: legacy, prefer `Image.add_local_*`

For input concurrency, use the *separate* decorator: `@modal.concurrent(max_inputs=N, target_inputs=M)`.

### Function methods
- `f.remote(*a, **kw)`: call & wait
- `f.local(*a, **kw)`: call locally
- `f.spawn(*a, **kw)`: fire-and-forget → `FunctionCall` (max 1M pending)
- `f.map(iter, return_exceptions=False, order_outputs=True)`: parallel (max 1,000 concurrent)
- `f.starmap(iter)`, `f.for_each(iter)`, `f.spawn_map(iter)`, `f.remote_gen(...)`
- `f.get_current_stats()` → `FunctionStats(backlog, num_active_runners)`
- `f.update_autoscaler(min_containers=..., max_containers=..., scaledown_window=...)`: runtime override
- `Function.from_name(app_name, function_name, environment_name=None)`: cross-app

Caps: 2,000 pending, 25,000 total inputs, 1M `.spawn()` queue.

### Image
Factories:
- **`Image.debian_slim(python_version=None)`**: default; matches local Python minor
- **`Image.from_registry(url, secret=None, add_python=None, setup_dockerfile_commands=None)`**: must be `linux/amd64`
- `Image.from_dockerfile(path, add_python=None)`
- `Image.from_aws_ecr(url, secret=...)`: needs AWS_ACCESS_KEY_ID/SECRET/REGION or AWS_ROLE_ARN/REGION
- `Image.from_gcp_artifact_registry(url, secret=...)`: SERVICE_ACCOUNT_JSON
- `Image.micromamba(python_version=None)`, `Image.from_scratch()` (sandbox-only), `Image.from_id(id)`

Build steps (chainable, each = a layer):
- **`apt_install(*pkgs)`**, **`pip_install(*pkgs)`**, `pip_install_from_requirements(path)`, `pip_install_from_pyproject(path)`
- **`uv_pip_install(*pkgs)`**, `uv_sync(uv_project_dir)` (v1.1.0+, **faster: recommended**)
- `poetry_install_from_file(pyproject, lockfile=None)`, `micromamba_install(*pkgs)`
- **`run_commands(*shell_cmds)`**
- `run_function(fn, **kwargs)`: runs Python at build; rebuilds only on raw_f source/kwargs/referenced-globals changes (NOT nested fns)
- `dockerfile_commands(*lines)`, `env({"V":"x"})`, `workdir(p)`, `entrypoint(cmd)`, `shell(p)`
- **`add_local_file(local, remote, copy=False)`**
- **`add_local_dir(local, remote, copy=False, ignore=None)`**
- **`add_local_python_source(module, copy=False, ignore=None)`**: adds package to PYTHONPATH at `/root/{module}`
- `imports()`: context for top-level imports of remote-only packages
- `force_build=True` per layer to rebuild

`copy=False` (default) mounts at startup; `copy=True` bakes into image. Cache busters: `MODAL_FORCE_BUILD=1`, `MODAL_IGNORE_CACHE=1`.

### Volume
- **`Volume.from_name(name, environment=None, create_if_missing=False, version=None)`**: lazy ref
- `Volume.from_id(id)`, `Volume.ephemeral()` (context manager)
- Methods on instance: **`commit()`**, **`reload()`**, `listdir(path, recursive=False)`, `iterdir(...)`, `read_file(path)`, `remove_file(path)`, `copy_files(src_paths, dst)`, `batch_upload(local_dir, remote_dir, force=False)`, `rename(src, dst)`
- Mount: `volumes={"/mnt": vol}`, `read_only=True` supported

### Secret
- **`Secret.from_name(name, environment_name=None)`**
- `Secret.from_dict({"K": "V"})` (None values dropped)
- `Secret.from_dotenv(path=None, filename=None)` (cwd default)
- `Secret.from_local_environ([...])`

Limits: keys ≤16 KiB alphanum+underscore (no leading digit); values ≤32 KiB. Larger blobs → use Volumes.

### Cls (stateful)
```python
@app.cls(gpu="A100", scaledown_window=300, enable_memory_snapshot=True)
class Predictor:
    model_name: str = modal.parameter()         # parametrize instances

    @modal.enter(snap=True)                     # runs BEFORE memory snapshot
    def load(self):
        self.model = load(self.model_name)

    @modal.enter(snap=False)                    # runs AFTER restore (e.g. .cuda())
    def warm(self):
        self.model.cuda()

    @modal.method()
    def infer(self, x): return self.model(x)

    @modal.exit()                               # 30 s grace before kill
    def cleanup(self): ...

    @modal.batched(max_batch_size=8, wait_ms=50)
    def batch_infer(self, xs): ...
```
Runtime: `Predictor(model_name="m").infer.remote(x)`. Class-level: `Predictor.with_options(gpu="H100").infer.remote(x)`: overrides only, can't unset. `Predictor.with_concurrency(...)`, `Predictor.with_batching(...)`, `Cls.from_name(...)`.

### Sandbox (one-off arbitrary commands)
```python
sb = modal.Sandbox.create(
    "python", "-c", "import torch; print(torch.cuda.is_available())",
    app=app, gpu="any", image=img, timeout=300,
    volumes=..., secrets=..., region=...,
)
sb.wait()
print(sb.stdout.read())
sb.terminate()
```
Default timeout **5 min**, max 24 h. Methods: `exec(*cmd)` → `ContainerProcess`, `terminate()`, `wait()`, `wait_until_ready()`, `poll()`, `detach()`. Streams: `.stdin/.stdout/.stderr/.returncode`. Filesystem: `sb.filesystem.{read,write}_{bytes,text}`, `copy_to_local`, `copy_from_local`, `remove`, `make_directory`. Supports tags, named sandboxes, readiness probes, FS snapshots.

### Web endpoints
```python
@app.function()
@modal.fastapi_endpoint(method="POST")
def api(payload: dict): ...                    # returns JSON

@app.function()
@modal.asgi_app()
def serve(): return fastapi_app                # full ASGI

@app.function()
@modal.wsgi_app()
def serve(): return flask_app                  # full WSGI

@app.function()
@modal.web_server(port=8000)                   # arbitrary HTTP server: MUST bind 0.0.0.0
def run(): subprocess.Popen([...])
```
Body limit 4 GiB; responses unlimited; WebSockets supported. Use `@modal.concurrent(max_inputs=N)` for parallel requests/container.

### `modal serve`
Hot-reload for web endpoints during dev (`modal serve app.py`).

## 5. GPU + pricing ladder

| GPU | $/sec | Notes |
|---|---|---|
| T4 | 0.000164 | 16 GB, cheapest |
| L4 | 0.000222 | 24 GB |
| A10 | 0.000306 | 24 GB; max 4/container |
| L40S | 0.000542 | 48 GB; great inference $/perf |
| A100-40GB | 0.000583 | |
| A100-80GB | 0.000694 | |
| H100 | 0.001097 | may auto-upgrade to H200; suffix `!` to forbid |
| H200 | 0.001261 | |
| B200 | 0.001736 | flagship; `B200+` allows B300 (billed as B200) |
| RTX-PRO-6000 | listed | |

CPU $0.0000131/core/s (min 0.125), memory $0.00000222/GiB/s. Free tier: **$30/mo credits**, 100 containers, 10 GPU concurrency.

Per-hour quick math: A100-40GB ≈ $2.10/h, A100-80GB ≈ $2.50/h, H100 ≈ $3.95/h, H200 ≈ $4.54/h, T4 ≈ $0.59/h. Multiply by GPU count for `:N`.

**Region multipliers:** US/EU/UK/AP **1.25×**, CA/SA/ME/MX/AF **2.5×**: pin carefully.

## 6. CUDA setup

Pre-installed driver **580.95.05**, CUDA driver API **13.0**, plus `nvidia-smi`. Most cases: just `pip_install("torch")` on `Image.debian_slim()`. Need full CUDA toolkit (TensorRT-LLM, custom kernels): `Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")`. Choose CUDA ≤ host (12.* and 13.* guaranteed compatible).

## 7. Gotchas (will burn you)

- **Volume path prefix.** Writing `/foo.txt` writes to local container fs, **not** the volume. Always use the mount path (`/mnt/data/foo.txt`).
- **`vol.commit()` is needed for synchronous visibility.** Auto-commit fires every few seconds and on shutdown, but other containers won't see writes until then. Call `commit()` after important writes (checkpoints).
- **`vol.reload()` fails if files are open** ("busy volume"). Close handles before reloading.
- **Volume v1 limits:** 500K inodes hard, 50K recommended; >50K → linear latency growth. Frontend download cap 16 MB. v2 (beta) lifts these.
- **Extraction policy: never extract large datasets onto a v1 volume.** Each frame, mask, or flow file = 1 inode. A typical stereo / video dataset (Scene Flow extracted ≈ 80K files, KITTI ≈ 50K, Middlebury ≈ 5K) blows past the 50K recommended threshold fast. Keep tarballs/zips compressed on the volume; extract inside the training container's local disk on demand (or on a scratch path that isn't the volume). If extraction on the volume is unavoidable, switch to v2: `Volume.from_name("name", create_if_missing=True, version=2)`.
- **Last-write-wins, no file locking.** Concurrent writers to same file lose data. v1 ≤5 concurrent writers, v2 hundreds.
- **`statfs` lies on volumes.** `df`/`shutil.disk_usage()` return placeholders. Use `du`.
- **Image cache cascade.** Breaking layer N rebuilds N..end. Put fast-changing layers (code) LAST.
- **`run_function` rebuild rule.** Triggers only on raw_f source, kwargs, and referenced globals: NOT nested function bodies.
- **gRPC payload limit 100 MB** → "413 Content Too Large". Pass via Volume for large blobs.
- **GIL holds break heartbeat → container kill.** Run blocking native code in subprocess. Debug with `py-spy` (preinstalled in `modal shell`).
- **Container reuse side effects.** SQLite "table already exists", port collisions. Idempotent code, randomized filenames, or last-resort `single_use_containers=True` (slow, costly).
- **Forked processes inherit stale Modal client state.** Recreate Modal client after fork (Celery, multiprocessing). 
- **L4 CUDA init flake** on some hosts: known issue.
- **Memory Snapshot pitfalls.** Multi-GPU code largely incompatible with GPU snapshots. Doesn't speed I/O, only imports/JIT. Randomness state captured (verify resilience). `torch.compile` may fail snapshot creation (set `TORCHINDUCTOR_COMPILE_THREADS=1`). xformers needs `XFORMERS_ENABLE_TRITON=1`.
- **Period resets on redeploy**, Cron does not.
- **Web server must bind `0.0.0.0`**, never `127.0.0.1`.
- **Region multipliers**: don't pin CA/SA/ME/MX/AF without realizing 2.5× cost.
- **Control plane is us-east-1** even for compute in eu/ap → input/output latency.
- **`modal: command not found`** → use `python -m modal` or fix PATH.
- **`modal container exec`** needs `--` before flagged sub-cmd: `modal container exec CID -- bash -lc "..."`.
- **`Mount` class is deprecated** → use `Image.add_local_*` instead.
- **`--detach` is the only way to survive disconnect.** Without it, `modal run` kills the app when the laptop sleeps.
- **Timeouts may overshoot by "a handful of seconds"**: not for hard real-time.
- **`with_options` cannot unset**: only override.
- **`add_local_dir(".", ...)`** ships your venv. Use `ignore=` to skip `.venv`, `.git`, `__pycache__`, `*.pyc`, etc., or rely on `pip_install` inside the image.

## 8. Cost optimization

- **Right-size GPU.** L40S ≈ A100-40GB perf at lower cost for inference. T4/L4 for batch CPU-light work. Don't reach for H100/B200 unless memory-bound.
- **Tune `scaledown_window`.** Default 60 s. Bursty cheap-warmup → 5-10 s. Expensive load → 300+ s.
- **Memory snapshots** + `@modal.enter(snap=True)` → 3-10× faster cold starts → fewer warm containers needed.
- **`min_containers=0`** for true scale-to-zero. Use `buffer_containers` for bursts.
- **Volumes for weights**, not bake-into-image: image rebuilds re-download otherwise.
- **`@modal.enter` to load weights once** per container, reused across calls.
- **`@modal.concurrent(max_inputs=N)`** packs I/O-bound or vLLM batching onto fewer GPUs.
- **`uv_pip_install` over `pip_install`** for faster image builds.
- **`add_local_*(copy=False)`** mounts at start, no image rebuild on code edits.
- **Avoid CA/SA/ME/MX/AF regions** unless required (2.5× multiplier).
- **Don't run downloads on GPU containers.** Use a CPU container with the same Volume mounted.

## 9. Quick-recall command index

| Need | Command |
|---|---|
| Run once | `modal run file.py::fn` |
| Long detached run | `modal run -d file.py::fn` |
| Tail logs | `modal app logs NAME -f` |
| Recent log search | `modal app logs NAME --since 2h --search "loss"` |
| List apps | `modal app list` |
| Kill run | `modal app stop NAME -y` |
| Shell into env | `modal shell file.py::fn` |
| One-off GPU shell | `modal shell --gpu a100 -c "nvidia-smi"` |
| List volumes | `modal volume list` |
| List files in volume | `modal volume ls VOL [PATH]` |
| Pull file from volume | `modal volume get VOL REMOTE LOCAL` |
| Push file to volume | `modal volume put VOL LOCAL REMOTE -f` |
| Remove from volume | `modal volume rm VOL PATH -r` |
| New secret | `modal secret create NAME K=V` |
| Deploy persistent app | `modal deploy file.py --name NAME` |
| Rollback deploy | `modal app rollback NAME [VERSION]` |
| Switch profile | `modal profile activate NAME` |

## 10. Known doc gaps (be careful)

- `modal launch` and `modal nfs` 404'd in the doc fetch: likely renamed/deprecated. Don't rely on them.
- `modal.gpu.X(...)` constructor classes exist in older code/blogs but are not in the current API: use **string syntax** (`gpu="A100-40GB:2"`).
- Older code may use `container_idle_timeout=` instead of `scaledown_window=`. Both reportedly accepted; prefer `scaledown_window` for current API.
- Spot/preemptible GPU support: not documented in the current corpus: assume unavailable unless you find an explicit announcement.
