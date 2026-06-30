# Modal scripts

Modal apps for training StereoLite on the cloud. See the `modal-expert`
skill at `~/.claude/skills/modal-expert/SKILL.md` for the full reference.

## Master Volume

All datasets live on a single Modal Volume named `stereo-datasets`,
laid out as:

```
/data/
  sceneflow/
    driving/
      driving__frames_finalpass.tar
      driving__disparity.tar.bz2
    monkaa/
      monkaa__frames_finalpass.tar
      monkaa__disparity.tar.bz2
    flyingthings3d/
      flyingthings3d__frames_finalpass.tar
      flyingthings3d__disparity.tar.bz2
  eth3d/         (planned)
  middlebury/    (planned)
  kitti2012/     (planned, manual upload: registration-walled)
  kitti2015/     (planned, manual upload: registration-walled)
```

Tarballs are kept compressed on the volume. Extraction happens inside
training containers on local disk to avoid blowing past the volume's
inode limit.

## Resource sizing for downloads

`cpu=0.5, memory=512 MiB` per download container.
At $0.0000131/core/s + $0.00000222/GiB/s, that is roughly **$0.03/h**.
A full Scene Flow download (~12 h wall) costs ~**$0.36** total compute,
plus a few cents for the volume traffic.

## Workflow

**Concurrency model**: each launch downloads exactly **one subset**.
Within that subset, the 2 files (frames + disparity) run in parallel as
2 separate `download_one` containers. So at most 2 concurrent streams
to Freiburg per launch. You decide when to start the next subset by
launching again with a different `--subset`. Each `download_one` has
its own 24h timeout and 5 retries with backoff.

```bash
# Activate venv first
source venv/bin/activate

# === Run 1: driving (~16 GB, ~30 min) ===
modal run -d model/scripts/modal/download_sceneflow.py::main \
    --action download --subset driving

# === Run 2 (when driving is done): monkaa (~51 GB, ~1.5 h) ===
modal run -d model/scripts/modal/download_sceneflow.py::main \
    --action download --subset monkaa

# === Run 3 (when monkaa is done): flyingthings3d (~138 GB, ~4-8 h) ===
modal run -d model/scripts/modal/download_sceneflow.py::main \
    --action download --subset flyingthings3d

# Re-run a single failed file by index (0..5; FILES in the script):
modal run model/scripts/modal/download_sceneflow.py::download_one --idx 4

# Check progress anytime, from any terminal:
modal app list
modal app logs download-sceneflow --follow
modal volume ls stereo-datasets sceneflow

# Per-file detailed status (all 6 files across all subsets):
modal run model/scripts/modal/download_sceneflow.py::main --action status

# Cross-dataset status across the whole master volume:
modal run model/scripts/modal/status_datasets.py

# Stop a running download (kills the 2 in-flight downloads of that subset):
modal app stop download-sceneflow -y

# Delete a stale partial file from the volume:
modal volume rm stereo-datasets sceneflow/<subset>/<file>
```

`wget -c` resumes partial downloads on re-invocation; functions are
idempotent. `vol.commit()` runs after each file completes so other
containers (e.g. `status`) see progress without waiting for shutdown.

## Middlebury training on Modal

Download Middlebury once into the master dataset Volume:

```bash
modal run -d model/scripts/modal/download_middlebury.py::main \
    --action download --year all

modal app logs download-middlebury --follow
modal run model/scripts/modal/download_middlebury.py::main --action status
```

Launch GEV4 training on Modal. `--batch` is per GPU; for early/cheap
experiments prefer `T4`, `L4`, or `A10`. Multi-GPU only makes sense after
the single-GPU run is behaving well. The launcher uses Modal `.spawn()` by
default, so it submits the cloud job and exits; your PC does not need to stay
on after the command prints `Spawned Modal training job`.

```bash
modal run -d model/scripts/modal/train_middlebury.py::main \
    --gpu A10 \
    --run-name yolo_ctx_gev4_mb_a10_b4_10k \
    --steps 10000 \
    --batch 4 \
    --height 384 \
    --width 640 \
    --train-mode mixed \
    --crop-prob 0.25

modal app logs stereolite-middlebury-train --follow
modal volume ls stereolite-results middlebury_runs/yolo_ctx_gev4_mb_a10_b4_10k
```

If you want the terminal to wait until training finishes, add `--wait 1`.
For overnight runs, leave it as the default `--wait 0`.

Results live in the `stereolite-results` Volume under
`/middlebury_runs/<run-name>/`. Pull a checkpoint with:

```bash
modal volume get stereolite-results \
    /middlebury_runs/yolo_ctx_gev4_mb_a10_b4_10k/best.pth ./best.pth
```

## Probe scripts

`probe1_hello.py` / `probe2_volume.py` / `probe3_gpu.py` were used to
verify the Modal setup (auth, volumes persist across runs, GPUs are
per-invocation, `modal volume get` works). Keep for reference; safe to
delete once you trust the workflow.

## Planned next datasets

- **ETH3D two-view** (~750 MB): public, scriptable.
- **Middlebury 2014 / 2021** (~5-10 GB): public, scriptable.
- **KITTI 2012 / 2015** (~3-4 GB each): registration-walled. Will
  require local download then `modal volume put stereo-datasets ./kitti2015 /kitti2015 -f`.
