"""Download KITTI stereo 2012 + 2015 zips to the stereo-datasets volume.

Zips stay COMPRESSED on the volume (v1 inode budget rule); eval drivers
extract to container-local /tmp. Public AVG S3 mirrors, no registration
needed for the training archives:

    data_stereo_flow.zip   (KITTI 2012, ~2.0 GB, 194 training pairs)
    data_scene_flow.zip    (KITTI 2015, ~1.7 GB, 200 training pairs)

CPU-only container. Usage:
    modal run model/scripts/modal/download_kitti.py::main
"""
from __future__ import annotations

import modal

app = modal.App("download-kitti")
vol = modal.Volume.from_name("stereo-datasets")
img = modal.Image.debian_slim().apt_install("wget", "ca-certificates")

DEST_ROOT = "/data/kitti"
FILES = [
    ("data_stereo_flow.zip",
     "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip"),
    ("data_scene_flow.zip",
     "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip"),
]


@app.function(image=img, volumes={"/data": vol}, cpu=4, memory=8192,
              timeout=3 * 3600, retries=0)
def download() -> list:
    import os
    import subprocess
    import time
    from pathlib import Path

    Path(DEST_ROOT).mkdir(parents=True, exist_ok=True)
    report = []
    for fname, url in FILES:
        dest = f"{DEST_ROOT}/{fname}"
        if os.path.exists(dest) and os.path.getsize(dest) > 1e9:
            print(f"  {fname}: already present "
                  f"({os.path.getsize(dest)/1e9:.2f} GB), skip")
            report.append((fname, "cached", os.path.getsize(dest)))
            continue
        t0 = time.time()
        rc = subprocess.call(["wget", "-c", "-q", "-O", dest, url])
        dt = time.time() - t0
        size = os.path.getsize(dest) if os.path.exists(dest) else 0
        print(f"  {fname}: rc={rc}  {size/1e9:.2f} GB  {dt:.0f} s")
        if rc != 0 or size < 1e9:
            raise RuntimeError(f"{fname}: download failed (rc={rc}, {size} B)")
        report.append((fname, "downloaded", size))
    vol.commit()
    return report


@app.local_entrypoint()
def main():
    for fname, status, size in download.remote():
        print(f"{fname}: {status} ({size/1e9:.2f} GB)")
