"""One-time repack of the SceneFlow archives into fast-loading shards.

Why: the raw archives on `stereo-datasets` keep the inode budget safe but are
poison for training startup — streaming the FT3D disparity .tar.bz2 takes
112 min single-threaded (measured 2026-07-04). Every training container (and
every relaunch) would burn ~$6 of idle A100 on that. This job pays the cost
once on cheap CPU containers and writes ~90 large shards to a dedicated
volume `sceneflow-shards`; training containers then pull shards at full
sequential throughput and start in minutes.

Format (per shard, ~1.8 GB, pickle):
    [{"key": "<left png member name>",
      "left_png": bytes,          # original PNG bytes, no re-encode
      "right_png": bytes,
      "disp_z": bytes,            # zstd(float32 LE disparity, row-major)
      "shape": (H, W)}, ...]
Disparity stays float32: fp16 would quantize GT by up to 0.25 px at large
disparities, unacceptable for sub-pixel EPE targets. Images stay native
540x960; resolution remains a training-time choice.

Split membership comes from the committed manifest
(model/configs/sceneflow_split_v1.json.gz), so shards are byte-identical to
the canonical protocol: shard prefixes train_ft3d / train_monkaa /
train_driving / test_ft3d. An index.json on the volume maps shard -> keys.

Blocking .map(); do NOT `modal run -d`.

Usage:
    modal run model/scripts/modal/repack_sceneflow_shards.py::main
"""
from __future__ import annotations

import modal

app = modal.App("repack-sceneflow-shards")
data_vol = modal.Volume.from_name("stereo-datasets")
shards_vol = modal.Volume.from_name("sceneflow-shards", create_if_missing=True)
PROJECT_ROOT = "/home/abrar/Research/stero_research_claude"
MANIFEST = "model/configs/sceneflow_split_v1.json.gz"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("lbzip2")
    .pip_install("numpy<2", "zstandard")
    .add_local_file(f"{PROJECT_ROOT}/{MANIFEST}", "/workspace/split.json.gz")
)

SUBSETS = {
    "ft3d": {
        "frames": "sceneflow/flyingthings3d/flyingthings3d__frames_finalpass.tar",
        "disp": "sceneflow/flyingthings3d/flyingthings3d__disparity.tar.bz2",
        "lists": [("train_ft3d", ("train", "ft3d_train")),
                  ("test_ft3d", ("test", None))],
    },
    "monkaa": {
        "frames": "sceneflow/monkaa/monkaa__frames_finalpass.tar",
        "disp": "sceneflow/monkaa/monkaa__disparity.tar.bz2",
        "lists": [("train_monkaa", ("train", "monkaa"))],
    },
    "driving": {
        "frames": "sceneflow/driving/driving__frames_finalpass.tar",
        "disp": "sceneflow/driving/driving__disparity.tar.bz2",
        "lists": [("train_driving", ("train", "driving"))],
    },
}
SHARD_BYTES = int(1.8e9)


def read_pfm_bytes(raw: bytes):
    import io

    import numpy as np
    f = io.BytesIO(raw)
    if f.readline().strip() != b"Pf":
        raise ValueError("not a grayscale PFM")
    w, h = map(int, f.readline().split())
    scale = float(f.readline())
    data = np.frombuffer(f.read(w * h * 4), "<f4" if scale < 0 else ">f4")
    return np.ascontiguousarray(np.flipud(data.reshape(h, w)).astype("<f4"))


@app.function(image=image, volumes={"/data": data_vol, "/shards": shards_vol},
              cpu=16, memory=32768, timeout=4 * 3600, retries=0)
def repack_subset(subset: str) -> dict:
    import gzip
    import json
    import pickle
    import subprocess
    import tarfile
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import zstandard

    cfg = SUBSETS[subset]
    man = json.loads(gzip.decompress(Path("/workspace/split.json.gz").read_bytes()))

    t0 = time.time()
    cctx = zstandard.ZstdCompressor(level=3)

    # 1) Parallel-decompress the disparity archive (lbzip2, all cores) and
    # stream-extract LEFT maps only, parsed + zstd'd on the fly. Names are
    # normalized (leading './' stripped) so they match the manifest exactly.
    # Disk holds ~27 GB of compressed float32 instead of 55 GB of raw PFM.
    disp_root = Path(f"/tmp/{subset}_disp")
    disp_root.mkdir(parents=True, exist_ok=True)
    n_disp = 0
    proc = subprocess.Popen(["lbzip2", "-dc", f"/data/{cfg['disp']}"],
                            stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as dtf:
        for m in dtf:
            if not m.isfile():
                continue
            name = m.name.lstrip("./")
            if "/left/" not in name or not name.endswith(".pfm"):
                continue
            d = read_pfm_bytes(dtf.extractfile(m).read())
            target = disp_root / (name + ".zst")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(cctx.compress(d.tobytes())
                               + d.shape[0].to_bytes(4, "little")
                               + d.shape[1].to_bytes(4, "little"))
            n_disp += 1
            if n_disp % 5000 == 0:
                print(f"[{subset}] {n_disp} disparity maps, "
                      f"{time.time()-t0:.0f}s")
    if proc.wait() != 0:
        raise RuntimeError(f"lbzip2 failed for {cfg['disp']}")
    print(f"[{subset}] {n_disp} left disparity maps ready in "
          f"{time.time()-t0:.0f}s")

    # 2) Index the frames tar once (header walk), then random-access members.
    tf = tarfile.open(f"/data/{cfg['frames']}", "r:")
    members = {m.name.lstrip("./"): m for m in tf.getmembers() if m.isfile()}
    tar_lock = threading.Lock()
    print(f"[{subset}] frames tar indexed: {len(members)} members, "
          f"{time.time()-t0:.0f}s")

    def build_record(left: str) -> dict:
        right = left.replace("/left/", "/right/")
        disp_rel = (left.replace("frames_finalpass", "disparity")
                        .replace(".png", ".pfm"))
        with tar_lock:
            lb = tf.extractfile(members[left]).read()
            rb = tf.extractfile(members[right]).read()
        blob = (disp_root / (disp_rel + ".zst")).read_bytes()
        h = int.from_bytes(blob[-8:-4], "little")
        w = int.from_bytes(blob[-4:], "little")
        return {"key": left, "left_png": lb, "right_png": rb,
                "disp_z": blob[:-8], "shape": (h, w)}

    # 3) Stream pairs -> shards.
    out_dir = Path("/shards/v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    index, n_done = [], 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for prefix, (split, sub) in cfg["lists"]:
            pairs = man[split][sub] if sub else man[split]
            buf, buf_bytes, shard_id = [], 0, 0

            def flush():
                nonlocal buf, buf_bytes, shard_id
                if not buf:
                    return
                name = f"{prefix}_{shard_id:04d}.pkl"
                with open(out_dir / name, "wb") as f:
                    pickle.dump(buf, f, protocol=4)
                index.append({"shard": name, "n": len(buf),
                              "bytes": buf_bytes,
                              "keys": [r["key"] for r in buf]})
                print(f"[{subset}] wrote {name}: {len(buf)} pairs, "
                      f"{buf_bytes/1e9:.2f} GB, {time.time()-t0:.0f}s")
                buf, buf_bytes, shard_id = [], 0, shard_id + 1

            for rec in ex.map(build_record, pairs, chunksize=8):
                buf.append(rec)
                buf_bytes += (len(rec["left_png"]) + len(rec["right_png"])
                              + len(rec["disp_z"]))
                n_done += 1
                if buf_bytes >= SHARD_BYTES:
                    flush()
                if n_done % 2000 == 0:
                    print(f"[{subset}] {n_done} pairs, {time.time()-t0:.0f}s")
                    shards_vol.commit()
            flush()
    tf.close()
    shards_vol.commit()
    total_gb = sum(e["bytes"] for e in index) / 1e9
    print(f"[{subset}] DONE: {n_done} pairs, {len(index)} shards, "
          f"{total_gb:.1f} GB in {time.time()-t0:.0f}s")
    return {"subset": subset, "pairs": n_done, "index": index}


@app.function(image=image, volumes={"/shards": shards_vol}, cpu=2,
              timeout=600, retries=0)
def write_index(index: dict):
    import json
    from pathlib import Path
    Path("/shards/v1/index.json").write_text(json.dumps(index, indent=1))
    shards_vol.commit()


@app.local_entrypoint()
def main():
    results = list(repack_subset.map(list(SUBSETS)))
    index = {"version": "v1", "manifest": "sceneflow_split_v1",
             "shards": [e for r in results for e in r["index"]]}
    counts = {r["subset"]: r["pairs"] for r in results}
    print(f"pair counts: {counts}")
    expected = {"ft3d": 22390 + 4370, "monkaa": 8664, "driving": 4400}
    for k, v in expected.items():
        print(f"  {k}: {counts.get(k)} vs {v} "
              f"[{'OK' if counts.get(k) == v else 'MISMATCH'}]")
    write_index.remote(index)
    n_train = sum(e["n"] for e in index["shards"]
                  if e["shard"].startswith("train_"))
    n_test = sum(e["n"] for e in index["shards"]
                 if e["shard"].startswith("test_"))
    print(f"index written: {len(index['shards'])} shards, "
          f"train={n_train}, test={n_test}")
