"""Cross-dataset status on the master `stereo-datasets` volume.

Walks /data/{sceneflow,eth3d,middlebury,kitti2012,kitti2015} and reports
disk usage + file count per subdir. Useful before launching a training
job to see what's actually present.

Run:
    modal run model/scripts/modal/status_datasets.py
"""
from __future__ import annotations

import modal

vol = modal.Volume.from_name("stereo-datasets", create_if_missing=True)
img = modal.Image.debian_slim()

app = modal.App("stereo-datasets-status")


@app.function(image=img, volumes={"/data": vol}, cpu=0.25, memory=256, timeout=300)
def report():
    import os
    vol.reload()
    if not os.path.isdir("/data"):
        return "(volume empty)"
    lines = [f"{'path':<40s}  {'files':>8s}  {'size_GB':>10s}"]
    lines.append("-" * 64)
    grand_total = 0
    for top in sorted(os.listdir("/data")):
        top_path = f"/data/{top}"
        if not os.path.isdir(top_path):
            continue
        nfiles, total = 0, 0
        for root, _, files in os.walk(top_path):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    total += os.path.getsize(fp)
                    nfiles += 1
                except OSError:
                    pass
        grand_total += total
        lines.append(f"{top+'/':<40s}  {nfiles:>8d}  {total/1e9:>10.2f}")
    lines.append("-" * 64)
    lines.append(f"{'TOTAL':<40s}  {'':>8s}  {grand_total/1e9:>10.2f}")
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(report.remote())
