"""Middlebury Stereo 2014 + 2021 downloader (Modal).

- 2021: single all.zip (~404 MiB, 24 scenes).
- 2014: 23 scenes × 2 calibration variants (perfect + imperfect)
        = 46 zips, ~5.6 GB total.

Saved to /data/middlebury/{2014,2021}/ on the master `stereo-datasets`
volume. Concurrency capped at 4 parallel downloads via max_containers
on download_one (polite to vision.middlebury.edu, fast enough overall).

Run:
    # 2021 only (404 MB, ~2 min)
    modal run -d model/scripts/modal/download_middlebury.py::main \\
        --action download --year 2021

    # 2014 only (5.6 GB, ~10-15 min with 4-way parallel)
    modal run -d model/scripts/modal/download_middlebury.py::main \\
        --action download --year 2014

    # both
    modal run -d model/scripts/modal/download_middlebury.py::main \\
        --action download --year all

    modal run    model/scripts/modal/download_middlebury.py::main --action status
"""
from __future__ import annotations

import modal

VOLUME_NAME = "stereo-datasets"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
img = modal.Image.debian_slim().apt_install("wget", "ca-certificates")

DEST_ROOT = "/data/middlebury"

SCENES_2014 = [
    "Adirondack", "Backpack", "Bicycle1", "Cable", "Classroom1", "Couch",
    "Flowers", "Jadeplant", "Mask", "Motorcycle", "Piano", "Pipes",
    "Playroom", "Playtable", "Recycle", "Shelves", "Shopvac", "Sticks",
    "Storage", "Sword1", "Sword2", "Umbrella", "Vintage",
]

# (year, url, size_min, size_max).
FILES = []

FILES.append((
    "2021",
    "https://vision.middlebury.edu/stereo/data/scenes2021/zip/all.zip",
    400_000_000, 450_000_000,
))

# Per-scene 2014 zips. Sizes from directory listing:
#   imperfect: 54-72 MB,  perfect: 77-104 MB.
# Bound 30-140 MB covers both with margin.
for scene in SCENES_2014:
    for variant in ("perfect", "imperfect"):
        FILES.append((
            "2014",
            f"https://vision.middlebury.edu/stereo/data/scenes2014/zip/"
            f"{scene}-{variant}.zip",
            30_000_000, 140_000_000,
        ))

YEARS = ("2014", "2021")

app = modal.App("download-middlebury")


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
    timeout=1800,
    max_containers=4,
    retries=modal.Retries(max_retries=5, initial_delay=60.0,
                          backoff_coefficient=2.0),
)
def download_one(idx: int):
    import os, subprocess, time
    if not 0 <= idx < len(FILES):
        raise ValueError(f"idx must be 0..{len(FILES)-1}, got {idx}")
    year, url, lo, hi = FILES[idx]
    target_dir = f"{DEST_ROOT}/{year}"
    os.makedirs(target_dir, exist_ok=True)
    fname = url.rsplit("/", 1)[-1]
    out = f"{target_dir}/{fname}"

    pre = _check_size(out, lo, hi)
    print(f"[{idx}] {year}/{fname}")
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
    lines = []
    grand_total = 0
    for year in YEARS:
        files_for_year = [(i, f) for i, f in enumerate(FILES) if f[0] == year]
        if not files_for_year:
            continue
        lines.append(f"\n=== {year} ===")
        ok, partial, missing, total = 0, 0, 0, 0
        for idx, (_, url, lo, hi) in files_for_year:
            fname = url.rsplit("/", 1)[-1]
            p = f"{DEST_ROOT}/{year}/{fname}"
            st = _check_size(p, lo, hi)
            sz = os.path.getsize(p) if os.path.exists(p) else 0
            total += sz
            if st.startswith("OK"):
                ok += 1
            elif st.startswith("missing"):
                missing += 1
            else:
                partial += 1
        lines.append(
            f"  {len(files_for_year)} files: "
            f"{ok} OK, {partial} partial, {missing} missing, "
            f"total {total/1e9:.2f} GB"
        )
        grand_total += total
    lines.append(f"\nGRAND TOTAL: {grand_total/1e9:.2f} GB")
    return "\n".join(lines)


@app.local_entrypoint()
def main(action: str = "status", year: str = ""):
    if action == "status":
        print(status.remote())
        return
    if action != "download":
        print(f"action must be 'download' or 'status', got {action!r}")
        return
    if year == "all":
        targets = list(YEARS)
    elif year in YEARS:
        targets = [year]
    else:
        print(f"--year must be {list(YEARS)} or 'all', got {year!r}")
        return
    indices = [i for i, (y, *_) in enumerate(FILES) if y in targets]
    print(f"=== middlebury {targets}: {len(indices)} files, "
          f"max 4 concurrent ===")
    for i in indices:
        download_one.spawn(i)
    print(f"spawned {len(indices)} calls. monitor:")
    print(f"  modal app logs download-middlebury --follow")
    print(f"  modal run model/scripts/modal/download_middlebury.py::main --action status")
