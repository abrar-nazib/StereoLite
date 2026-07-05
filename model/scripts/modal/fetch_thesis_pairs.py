"""Fetch a handful of full stereo records (left, right, disparity) from the
SceneFlow shard volume for thesis figure thumbnails.

The local run mirror stores left + GT + predictions for tracked scenes but
not the RIGHT image; thesis architecture figures need a genuine right view.
CPU-only, seconds of compute.

Shard record schema (see train_full_sceneflow.decode_record):
  rec["left_png"] / rec["right_png"]  raw PNG bytes (native 960x540)
  rec["disp_z"]                       zstd-compressed <f4 buffer
  rec["shape"]                        (h, w)
  rec["key"]                          e.g. "ft3d/TEST/A/0000/left/0009.png"
  seq = "/".join(key.split("/")[1:-2]), t = int(basename minus .png)

Usage:
    modal run model/scripts/modal/fetch_thesis_pairs.py::main
    modal volume get widener-results thesis_assets model/benchmarks/thesis_assets
"""
from __future__ import annotations

import modal

app = modal.App("fetch-thesis-pairs")
shards_vol = modal.Volume.from_name("sceneflow-shards")
results_vol = modal.Volume.from_name("widener-results")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy<2", "zstandard")
)

# (seq, t) of the tracked val scenes we want full pairs for
WANTED = {("TEST/A/0000", 9), ("TEST/A/0001", 13),
          ("TEST/A/0006", 14), ("TEST/A/0013", 6)}


@app.function(image=image,
              volumes={"/shards": shards_vol, "/results": results_vol},
              cpu=4, memory=16384, timeout=1800)
def fetch() -> list[str]:
    import pickle
    from pathlib import Path

    import numpy as np
    import zstandard

    out_dir = Path("/results/thesis_assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    remaining = set(WANTED)
    saved = []

    for sp in sorted(Path("/shards/v1").glob("test_ft3d_*.pkl")):
        if not remaining:
            break
        with open(sp, "rb") as f:
            recs = pickle.load(f)
        for r in recs:
            parts = r["key"].split("/")
            seq = "/".join(parts[1:-2])
            t = int(parts[-1][:-4])
            if (seq, t) not in remaining:
                continue
            tag = seq.replace("/", "_") + f"_t{t:02d}"
            (out_dir / f"{tag}_left.png").write_bytes(r["left_png"])
            (out_dir / f"{tag}_right.png").write_bytes(r["right_png"])
            h, w = r["shape"]
            d = np.abs(np.frombuffer(zstandard.decompress(r["disp_z"]),
                                     "<f4").reshape(h, w))
            np.save(out_dir / f"{tag}_disp.npy",
                    np.nan_to_num(d, nan=0.0, posinf=0.0))
            saved.append(tag)
            remaining.discard((seq, t))
            print(f"saved {tag} from {sp.name}", flush=True)
        del recs
    results_vol.commit()
    if remaining:
        print(f"NOT FOUND: {remaining}")
    return saved


@app.local_entrypoint()
def main():
    saved = fetch.remote()
    print("fetched:", saved)
    print("pull with: modal volume get widener-results thesis_assets "
          "model/benchmarks/thesis_assets")
