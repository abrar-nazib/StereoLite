"""ETH3D two-view (low-res) downloader (Modal).

3 small files, ~40 MB total. Saved to /data/eth3d/ on the master
`stereo-datasets` volume.

Run:
    modal run -d model/scripts/modal/download_eth3d.py::main --action download
    modal run    model/scripts/modal/download_eth3d.py::main --action status

Sizes from www.eth3d.net HEAD content-length (verified 2026-04-30):
    two_view_training.7z       13.6 MB
    two_view_training_gt.7z    14.2 MB
    two_view_test.7z           11.8 MB
"""
from __future__ import annotations

import modal

VOLUME_NAME = "stereo-datasets"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
img = modal.Image.debian_slim().apt_install("wget", "ca-certificates")

DEST_ROOT = "/data/eth3d"

# (kind, url, size_min, size_max). Bounds ~10% wider than HEAD reported.
FILES = [
    ("training",
     "https://www.eth3d.net/data/two_view_training.7z",
     12_000_000, 16_000_000),                                  # 0
    ("training_gt",
     "https://www.eth3d.net/data/two_view_training_gt.7z",
     13_000_000, 17_000_000),                                  # 1
    ("test",
     "https://www.eth3d.net/data/two_view_test.7z",
     10_500_000, 14_000_000),                                  # 2
]

app = modal.App("download-eth3d")


def _check_size(path: str, lo: int, hi: int) -> str:
    import os
    if not os.path.exists(path):
        return "missing"
    sz = os.path.getsize(path)
    if lo <= sz <= hi * 1.1:
        return f"OK ({sz/1e6:.2f} MB)"
    if sz < lo:
        return f"partial ({100*sz/lo:.1f}% of {lo/1e6:.1f} MB)"
    return f"OVERSIZE ({sz/1e6:.2f} MB > {hi/1e6:.1f} MB)"


@app.function(
    image=img,
    volumes={"/data": vol},
    cpu=0.25,
    memory=256,
    timeout=600,
    retries=modal.Retries(max_retries=5, initial_delay=60.0,
                          backoff_coefficient=2.0),
)
def download_one(idx: int):
    import os, subprocess, time
    if not 0 <= idx < len(FILES):
        raise ValueError(f"idx must be 0..{len(FILES)-1}, got {idx}")
    kind, url, lo, hi = FILES[idx]
    target_dir = f"{DEST_ROOT}"
    os.makedirs(target_dir, exist_ok=True)
    fname = url.rsplit("/", 1)[-1]
    out = f"{target_dir}/{fname}"

    pre = _check_size(out, lo, hi)
    print(f"[{idx}] {fname}")
    print(f"  pre: {pre}")
    if pre.startswith("OK"):
        return {"idx": idx, "status": "already_complete"}

    t0 = time.time()
    rc = subprocess.run([
        "wget", "-c",
        "--tries=0",
        "--waitretry=30",
        "--timeout=120",
        "--read-timeout=300",
        "--retry-connrefused",
        "--progress=dot:mega",
        "--directory-prefix", target_dir,
        url,
    ]).returncode
    dt = time.time() - t0
    post = _check_size(out, lo, hi)
    print(f"  post: {post}  (wget rc={rc}, {dt:.1f} s)")
    vol.commit()
    if not post.startswith("OK"):
        raise RuntimeError(f"{fname}: {post} (wget rc={rc})")
    return {"idx": idx, "status": "downloaded", "seconds": dt}


@app.function(image=img, volumes={"/data": vol},
              cpu=0.25, memory=128, timeout=120)
def status():
    import os
    vol.reload()
    if not os.path.isdir(DEST_ROOT):
        return f"(no {DEST_ROOT} yet)"
    lines = [f"{'idx  file':<50s}  {'status':<35s}  {'MB':>8s}"]
    lines.append("-" * 100)
    total = 0
    for idx, (_kind, url, lo, hi) in enumerate(FILES):
        fname = url.rsplit("/", 1)[-1]
        p = f"{DEST_ROOT}/{fname}"
        st = _check_size(p, lo, hi)
        sz = os.path.getsize(p) if os.path.exists(p) else 0
        total += sz
        lines.append(f"{idx:<3d}  {fname:<45s}  {st:<35s}  {sz/1e6:>8.2f}")
    lines.append("-" * 100)
    lines.append(f"{'TOTAL':<50s}  {'':<35s}  {total/1e6:>8.2f}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(action: str = "status"):
    if action == "status":
        print(status.remote())
        return
    if action != "download":
        print(f"action must be 'download' or 'status', got {action!r}")
        return
    print(f"=== ETH3D: spawning {len(FILES)} parallel downloads ===")
    for i in range(len(FILES)):
        fname = FILES[i][1].rsplit("/", 1)[-1]
        h = download_one.spawn(i)
        print(f"  [{i}] {fname}  ->  call_id={h.object_id}")
    print(f"\nspawned. monitor:")
    print(f"  modal app logs download-eth3d --follow")
    print(f"  modal run model/scripts/modal/download_eth3d.py::main --action status")
