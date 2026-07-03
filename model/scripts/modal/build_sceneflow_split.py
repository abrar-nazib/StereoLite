"""Build the canonical SceneFlow train/test split manifest from the raw
archives on the `stereo-datasets` Modal volume.

Protocol (standard since PSMNet, confirmed by OpenStereo / IGEV):
  train = FlyingThings3D TRAIN + Monkaa (all) + Driving (all)  -> 35,454 pairs
  test  = FlyingThings3D TEST                                  -> 4,370 pairs
  images: frames_finalpass; eval metric: EPE, valid mask disparity < 192.

There is no separate validation split in the literature; papers report EPE
directly on the FT3D TEST set. For cheap periodic validation during training
we additionally emit `val_subset`: a fixed seeded 400-pair sample of the test
set (full 4,370 evaluated at the end / at milestones).

The archives are scanned WITHOUT extraction (tar header walk for .tar,
streaming decode for .tar.bz2), so the volume's inode budget is untouched.
Six CPU-only containers run in parallel, one per archive.

Output:
  local:  model/data/sceneflow_split_v1.json.gz   (checked into the repo)
  volume: widener-results:/sceneflow_split/sceneflow_split_v1.json.gz

Blocking .map(); do NOT `modal run -d`.

Usage:
    modal run model/scripts/modal/build_sceneflow_split.py::main
"""
from __future__ import annotations

import modal

app = modal.App("build-sceneflow-split")
data_vol = modal.Volume.from_name("stereo-datasets")
results_vol = modal.Volume.from_name("widener-results", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12")

ARCHIVES = {
    "ft3d_frames": "sceneflow/flyingthings3d/flyingthings3d__frames_finalpass.tar",
    "ft3d_disp": "sceneflow/flyingthings3d/flyingthings3d__disparity.tar.bz2",
    "monkaa_frames": "sceneflow/monkaa/monkaa__frames_finalpass.tar",
    "monkaa_disp": "sceneflow/monkaa/monkaa__disparity.tar.bz2",
    "driving_frames": "sceneflow/driving/driving__frames_finalpass.tar",
    "driving_disp": "sceneflow/driving/driving__disparity.tar.bz2",
}

# Official pair counts for the standard protocol.
EXPECTED = {"ft3d_train": 22390, "ft3d_test": 4370,
            "monkaa": 8664, "driving": 4400,
            "train_total": 35454, "test_total": 4370}


@app.function(image=image, volumes={"/data": data_vol}, cpu=4,
              timeout=4 * 3600, retries=0)
def scan_archive(key: str) -> dict:
    """Return every file member name in one archive."""
    import tarfile
    import time

    path = f"/data/{ARCHIVES[key]}"
    mode = "r|bz2" if path.endswith(".bz2") else "r:"
    names, t0 = [], time.time()
    with tarfile.open(path, mode) as tf:
        for m in tf:
            if m.isfile():
                names.append(m.name)
            if len(names) % 20000 == 0 and names:
                print(f"[{key}] {len(names)} members, {time.time()-t0:.0f}s")
    print(f"[{key}] DONE: {len(names)} file members in {time.time()-t0:.0f}s")
    return {"key": key, "names": names}


@app.function(image=image, volumes={"/results": results_vol}, cpu=2,
              timeout=600, retries=0)
def store_manifest(blob: bytes) -> str:
    from pathlib import Path
    out = Path("/results/sceneflow_split")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sceneflow_split_v1.json.gz").write_bytes(blob)
    results_vol.commit()
    return str(out / "sceneflow_split_v1.json.gz")


def norm(name: str) -> str:
    """Strip any leading './' and collapse the archive-root dir variants."""
    return name.lstrip("./")


def build_pairs(frames: set[str], disps: set[str], subset: str,
                want_prefix: str | None) -> tuple[list, list]:
    """Match left/right frames with left disparity; return (pairs, problems).

    A pair is recorded as the left-image member name; right image and
    disparity paths are derived deterministically:
      right = left with '/left/' -> '/right/'
      disp  = left with 'frames_finalpass' -> 'disparity', '.png' -> '.pfm'
    """
    pairs, problems = [], []
    for f in sorted(frames):
        if "/left/" not in f or not f.endswith(".png"):
            continue
        if want_prefix is not None and want_prefix not in f:
            continue
        right = f.replace("/left/", "/right/")
        disp = (f.replace("frames_finalpass", "disparity")
                 .replace(".png", ".pfm"))
        if right not in frames:
            problems.append(f"{subset}: missing right for {f}")
            continue
        if disp not in disps:
            problems.append(f"{subset}: missing disparity for {f}")
            continue
        pairs.append(f)
    return pairs, problems


@app.local_entrypoint()
def main():
    import gzip
    import hashlib
    import json
    import random
    from pathlib import Path

    print("scanning 6 archives in parallel (CPU containers) ...")
    scans = {r["key"]: {norm(n) for n in r["names"]}
             for r in scan_archive.map(list(ARCHIVES))}
    for k, v in scans.items():
        print(f"  {k}: {len(v)} file members")

    ft3d_train, p1 = build_pairs(scans["ft3d_frames"], scans["ft3d_disp"],
                                 "ft3d_train", "/TRAIN/")
    ft3d_test, p2 = build_pairs(scans["ft3d_frames"], scans["ft3d_disp"],
                                "ft3d_test", "/TEST/")
    monkaa, p3 = build_pairs(scans["monkaa_frames"], scans["monkaa_disp"],
                             "monkaa", None)
    driving, p4 = build_pairs(scans["driving_frames"], scans["driving_disp"],
                              "driving", None)
    problems = p1 + p2 + p3 + p4

    counts = {"ft3d_train": len(ft3d_train), "ft3d_test": len(ft3d_test),
              "monkaa": len(monkaa), "driving": len(driving),
              "train_total": len(ft3d_train) + len(monkaa) + len(driving),
              "test_total": len(ft3d_test)}
    print("\ncounts vs official protocol:")
    for k, v in counts.items():
        flag = "OK" if v == EXPECTED[k] else f"MISMATCH (expected {EXPECTED[k]})"
        print(f"  {k}: {v}  [{flag}]")
    if problems:
        print(f"\n{len(problems)} pairing problems (first 10):")
        for p in problems[:10]:
            print(f"  {p}")

    # Fixed seeded validation subset for cheap periodic eval during training.
    rng = random.Random(42)
    val_subset = sorted(rng.sample(ft3d_test, k=min(400, len(ft3d_test))))

    manifest = {
        "version": "v1",
        "protocol": {
            "train": "FlyingThings3D TRAIN + Monkaa (all) + Driving (all)",
            "test": "FlyingThings3D TEST (papers report EPE directly on this)",
            "pass": "finalpass",
            "eval_mask": "0 < disparity < 192 (standard)",
            "val_subset": "fixed seed-42 sample of 400 test pairs for "
                          "periodic validation; full test at milestones",
            "path_derivation": "right = left s|/left/|/right/|; disp = left "
                               "s|frames_finalpass|disparity| s|.png|.pfm|",
        },
        "archives": ARCHIVES,
        "counts": counts,
        "expected": EXPECTED,
        "problems": problems,
        "train": {"ft3d_train": ft3d_train, "monkaa": monkaa,
                  "driving": driving},
        "test": ft3d_test,
        "val_subset": val_subset,
    }
    raw = json.dumps(manifest, indent=1).encode()
    manifest["sha256_of_lists"] = hashlib.sha256(raw).hexdigest()
    blob = gzip.compress(json.dumps(manifest, indent=1).encode())

    local = Path(__file__).resolve().parents[2] / "data" / "sceneflow_split_v1.json.gz"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(blob)
    remote = store_manifest.remote(blob)
    print(f"\nwrote {local} ({len(blob)/1e6:.1f} MB) and {remote}")
    print(f"sha256(lists) = {manifest['sha256_of_lists']}")
