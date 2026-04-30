"""Scene Flow downloader (Modal).

One subset per launch. Within a subset, the 2 files (frames + disparity)
download in parallel as 2 separate containers. You decide when to start
the next subset by launching again with a different --subset.

Run:
    modal run -d model/scripts/modal/download_sceneflow.py::main \\
        --action download --subset driving
    modal run -d model/scripts/modal/download_sceneflow.py::main \\
        --action download --subset monkaa
    modal run -d model/scripts/modal/download_sceneflow.py::main \\
        --action download --subset flyingthings3d

    # status across all 6 files
    modal run    model/scripts/modal/download_sceneflow.py::main --action status

    # rerun a single failed file by index (0..5; see FILES below)
    modal run    model/scripts/modal/download_sceneflow.py::download_one --idx 4
"""
from __future__ import annotations

import modal

VOLUME_NAME = "stereo-datasets"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
img = modal.Image.debian_slim().apt_install("wget", "ca-certificates")

DEST_ROOT = "/data/sceneflow"
BASE = ("https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/"
        "Release_april16/data")

# (subset, url, expected_size_min, expected_size_max). Index = position in list.
# Sizes from Freiburg's official page
# (lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html):
#   driving:        frames 6.1 GB, disparity 9 GB
#   monkaa:         frames 17 GB,  disparity 28 GB
#   flyingthings3d: frames 43 GB,  disparity 87 GB
# Bounds below are ~10-15% wider than published to absorb minor variation.
FILES = [
    ("driving",
     f"{BASE}/Driving/raw_data/driving__frames_finalpass.tar",
     5_500_000_000, 7_500_000_000),                            # 0  (6.1 GB)
    ("driving",
     f"{BASE}/Driving/derived_data/driving__disparity.tar.bz2",
     8_000_000_000, 11_000_000_000),                           # 1  (9 GB)
    ("monkaa",
     f"{BASE}/Monkaa/raw_data/monkaa__frames_finalpass.tar",
     15_000_000_000, 20_000_000_000),                          # 2  (17 GB)
    ("monkaa",
     f"{BASE}/Monkaa/derived_data/monkaa__disparity.tar.bz2",
     25_000_000_000, 32_000_000_000),                          # 3  (28 GB)
    ("flyingthings3d",
     f"{BASE}/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar",
     40_000_000_000, 48_000_000_000),                          # 4  (43 GB)
    ("flyingthings3d",
     f"{BASE}/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2",
     82_000_000_000, 95_000_000_000),                          # 5  (87 GB)
]

SUBSETS = ["driving", "monkaa", "flyingthings3d"]

app = modal.App("download-sceneflow")


def _check_size(path: str, lo: int, hi: int) -> str:
    import os
    if not os.path.exists(path):
        return "missing"
    sz = os.path.getsize(path)
    if lo <= sz <= hi * 1.1:
        return f"OK ({sz/1e9:.2f} GB)"
    if sz < lo:
        return f"partial ({100*sz/lo:.1f}% of {lo/1e9:.1f} GB)"
    return f"OVERSIZE ({sz/1e9:.2f} GB > {hi/1e9:.1f} GB)"


@app.function(
    image=img,
    volumes={"/data": vol},
    cpu=0.5,
    memory=512,
    timeout=24 * 3600,
    retries=modal.Retries(max_retries=5, initial_delay=60.0,
                          backoff_coefficient=2.0),
)
def download_one(idx: int):
    """Download a single file by index into FILES. Resumes via wget -c."""
    import os, subprocess, time
    if not 0 <= idx < len(FILES):
        raise ValueError(f"idx must be 0..{len(FILES)-1}, got {idx}")
    subset, url, lo, hi = FILES[idx]
    target_dir = f"{DEST_ROOT}/{subset}"
    os.makedirs(target_dir, exist_ok=True)
    fname = url.rsplit("/", 1)[-1]
    out = f"{target_dir}/{fname}"

    pre = _check_size(out, lo, hi)
    print(f"[{idx}] {subset}/{fname}")
    print(f"  pre: {pre}")
    if pre.startswith("OK"):
        print(f"  already complete, skipping.")
        return {"idx": idx, "fname": fname, "status": "already_complete"}

    # NOTE: do NOT combine `-c` with `-O`. Per wget(1), `-O` overrides
    # `-c`, so on a restarted container the partial file gets truncated
    # and the download starts again from byte 0. Use `--directory-prefix`
    # + the URL's natural basename instead, which is what we want anyway
    # (URL ends in the filename we expect to write).
    t0 = time.time()
    rc = subprocess.run([
        "wget", "-c",
        "--tries=0",
        "--waitretry=30",
        "--timeout=120",
        "--read-timeout=300",
        "--retry-connrefused",
        "--progress=dot:giga",
        "--directory-prefix", target_dir,
        url,
    ]).returncode
    dt = time.time() - t0
    post = _check_size(out, lo, hi)
    print(f"  post: {post}  (wget rc={rc}, {dt/60:.1f} min)")

    vol.commit()
    if not post.startswith("OK"):
        raise RuntimeError(f"{fname}: {post} (wget rc={rc})")
    return {"idx": idx, "fname": fname, "status": "downloaded", "seconds": dt}


@app.function(image=img, volumes={"/data": vol},
              cpu=0.25, memory=128, timeout=120)
def status():
    import os
    vol.reload()
    if not os.path.isdir(DEST_ROOT):
        return f"(no {DEST_ROOT} yet: nothing downloaded)"
    lines = [f"{'idx  file':<60s}  {'status':<35s}  {'GB':>8s}"]
    lines.append("-" * 110)
    total = 0
    for idx, (subset, url, lo, hi) in enumerate(FILES):
        fname = url.rsplit("/", 1)[-1]
        p = f"{DEST_ROOT}/{subset}/{fname}"
        st = _check_size(p, lo, hi)
        sz = os.path.getsize(p) if os.path.exists(p) else 0
        total += sz
        lines.append(f"{idx:<3d}  {subset+'/'+fname:<55s}  {st:<35s}  {sz/1e9:>8.2f}")
    lines.append("-" * 110)
    lines.append(f"{'TOTAL':<60s}  {'':<35s}  {total/1e9:>8.2f}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(action: str = "status", subset: str = ""):
    if action == "status":
        print(status.remote())
        return
    if action != "download":
        print(f"action must be 'download' or 'status', got {action!r}")
        return
    if subset not in SUBSETS:
        print(f"--subset is required for download.\n"
              f"valid choices: {SUBSETS}\n"
              f"got: {subset!r}")
        return
    indices = [i for i, (s, *_) in enumerate(FILES) if s == subset]
    print(f"=== {subset}: spawning {len(indices)} parallel downloads ===")
    for i in indices:
        fname = FILES[i][1].rsplit("/", 1)[-1]
        h = download_one.spawn(i)
        print(f"  [{i}] {fname}  ->  call_id={h.object_id}")
    print(f"\nspawned. monitor with:")
    print(f"  modal app logs download-sceneflow --follow")
    print(f"  modal run model/scripts/modal/download_sceneflow.py::main --action status")
