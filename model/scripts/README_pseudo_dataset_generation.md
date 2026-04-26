# Generating pseudo-disparity dataset with FoundationStereo

This README is for **the person running the teacher model on a separate
PC** to produce pseudo-ground-truth disparity for stereo pairs. The
output `.npy` files are then sent back to be used as supervision when
training our smaller StereoLite model.

## What you are doing

1. You have a folder of **rectified stereo image pairs** (left + right
   PNG/JPG of the same scene).
2. You run the FoundationStereo teacher network (NVIDIA, CVPR 2025) on
   each pair.
3. Out comes one `.npy` disparity map per pair, plus a coloured `.png`
   visualisation for sanity check.
4. You bundle the output and send it back.

The teacher is heavy (~300 M parameters) and slow (1-3 s per pair on a
3060+, slower on smaller GPUs). The whole point of this exercise is to
generate dense, accurate pseudo-GT once on your beefier machine so we
don't have to run the teacher every time we train.

## Hardware requirements

| component | minimum | recommended |
|---|---|---|
| GPU VRAM | 6 GB | 12 GB+ (RTX 3060 or better) |
| Disk space | 2× the size of your input image folder | same |
| CPU | any modern x86_64 | — |
| OS | Linux (Ubuntu 22.04 tested) | — |
| CUDA | 11.8 or 12.x | 12.x |

If your GPU has < 8 GB VRAM, lower the inference resolution via the
`--inf_h` / `--inf_w` flags (defaults are 384 × 640).

## Setup (one-time)

```bash
# 1. Clone the project (the FoundationStereo teacher comes bundled
#    as part of model/teachers/)
git clone <project-repo-url> stereolite
cd stereolite

# 2. Create a Python 3.10+ environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch matching your CUDA version
#    (see https://pytorch.org/get-started/locally/ for the right wheel)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install FoundationStereo's runtime dependencies
pip install opencv-python numpy omegaconf einops timm scipy

# 5. (Optional but recommended) install xformers for faster ViT attention
pip install xformers
# If xformers fails to install, you can disable it via:
#   export XFORMERS_DISABLED=1
```

The FoundationStereo pretrained weights should already be present at:
```
model/teachers/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth
model/teachers/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
```
The `11-33-40` checkpoint is the **ViT-Small** variant (faster, ~6 GB
VRAM); the `23-51-11` checkpoint is the **ViT-Large** variant (slower,
~10 GB VRAM, slightly better quality). Default in `run_teacher.py` is
`11-33-40`. If you want better pseudo-GT and have the VRAM, point at the
larger one with `--ckpt`.

If the weights are missing, download them from the [official
FoundationStereo release](https://github.com/NVlabs/FoundationStereo).

## Input data layout

The script expects a flat directory with `left/` and `right/`
subdirectories, with **identical file names** in each:

```
data/pairs/
├── left/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── right/
    ├── 00000.png
    ├── 00001.png
    └── ...
```

Constraints:
- Pairs must be **rectified** (epipolar lines aligned to image rows). If
  your stereo camera produces a side-by-side combined frame, split it
  into `left/` and `right/` first.
- File names in `left/` and `right/` must match exactly.
- Image format: PNG or JPG, any resolution. The script will resize
  internally for inference.
- Recommended scene types: real-world driving, indoor, outdoor — the
  more diverse, the better the resulting student model.

## Running

From the project root:

```bash
source venv/bin/activate

python3 model/scripts/run_teacher.py \
    --pairs_dir data/pairs \
    --inf_h 384 --inf_w 640 \
    --save_h 384 --save_w 640 \
    --valid_iters 16
```

Common variants:

```bash
# Use the larger (better) ViT-Large checkpoint
python3 model/scripts/run_teacher.py \
    --pairs_dir data/pairs \
    --ckpt model/teachers/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth

# Higher input resolution (better disparity, more VRAM)
python3 model/scripts/run_teacher.py \
    --pairs_dir data/pairs \
    --inf_h 512 --inf_w 832 --save_h 512 --save_w 832

# Process only the first N pairs (for a quick test)
python3 model/scripts/run_teacher.py \
    --pairs_dir data/pairs --max_pairs 10
```

All flags:
| flag | default | meaning |
|---|---|---|
| `--pairs_dir` | `data/pairs` | directory containing `left/` and `right/` |
| `--ckpt` | `.../11-33-40/model_best_bp2.pth` | teacher checkpoint path |
| `--inf_h`, `--inf_w` | 384, 640 | resolution at which the teacher runs |
| `--save_h`, `--save_w` | 384, 640 | resolution at which disparity is saved |
| `--valid_iters` | 16 | refinement iterations inside the teacher (more = better, slower) |
| `--max_pairs` | (all) | cap the number of pairs processed |

## Output layout

The script writes alongside your `left/` and `right/` directories:

```
data/pairs/
├── left/                 # (your input)
├── right/                # (your input)
├── disp_pseudo/          # NEW — float32 disparity per pair
│   ├── 00000.npy
│   └── ...
├── disp_vis/             # NEW — TURBO-coloured visualisation
│   ├── 00000.png
│   └── ...
└── teacher_log.txt       # NEW — timing and per-pair stats
```

- `disp_pseudo/00000.npy`: 2D `float32` array of disparity values in
  pixels at `--save_h` × `--save_w` resolution. Load with
  `np.load("disp_pseudo/00000.npy")`.
- `disp_vis/00000.png`: TURBO colormap visualisation of the same
  disparity, for human sanity checking. Red = high disparity (close),
  blue = low (far).
- `teacher_log.txt`: per-pair latency, min/max/mean disparity, and the
  median throughput at the end.

## Sanity check

Before processing thousands of pairs, run on the first 10 and look at
the visualisations:

```bash
python3 model/scripts/run_teacher.py --pairs_dir data/pairs --max_pairs 10
xdg-open data/pairs/disp_vis/00000.png       # or use any image viewer
```

A reasonable disparity visualisation should:
- Have **smooth gradients** along surfaces (not noisy patches).
- Show **close objects in red/yellow** and **far objects in blue**.
- Have **sharp edges** at object boundaries.

If the output looks like noise or random colours: check that left and
right are not swapped, that they are rectified, and that file names
match between the two directories.

## Sending data back

The `.npy` files are the precious bit; the `disp_vis/` PNGs are
optional (they're just for QA).

```bash
# Bundle everything into one tarball
cd data/pairs
tar -cJf disp_pseudo_$(date +%Y%m%d).tar.xz \
    disp_pseudo/ teacher_log.txt
```

The `.tar.xz` should be ~15-30 % the size of the input PNGs (disparity
compresses well). For large datasets (>10 GB compressed) consider:
- A USB drive
- Resilio Sync / Syncthing
- A private cloud bucket (S3, GCS, R2)
- `scp -r` over a tunnelled SSH

Don't email it.

## Troubleshooting

**OOM (CUDA out of memory)**
- Lower `--inf_h` and `--inf_w` (try 256 × 512).
- Use the smaller `11-33-40` checkpoint instead of `23-51-11`.
- Lower `--valid_iters` (e.g. 8 instead of 16) — small accuracy hit, big
  speed/VRAM saving.

**`XFORMERS_DISABLED` warning or missing xformers**
- `export XFORMERS_DISABLED=1` before running. The model still works,
  it just uses slightly more memory.

**`No such file or directory: cfg.yaml`**
- Make sure the checkpoint directory contains a `cfg.yaml` next to the
  `.pth` file. Both `11-33-40/` and `23-51-11/` should already have one
  if you cloned the repo with submodules.

**Disparity output is all zero / all max**
- Left and right are probably swapped. Try renaming the directories.
- Or the images are not rectified. FoundationStereo assumes rectified
  input.

**Process very slow (>10 s per pair)**
- This is normal on consumer GPUs at the larger ViT size or higher
  resolution. Expect:
  - RTX 4090 + ViT-Small + 384x640: ~0.3 s/pair
  - RTX 3060 + ViT-Small + 384x640: ~1.5 s/pair
  - RTX 3060 + ViT-Large + 512x832: ~5 s/pair
- For 10k pairs at 1.5 s each = ~4 hours. Plan accordingly.

## Quick reference

```bash
# Most common command:
python3 model/scripts/run_teacher.py --pairs_dir data/pairs

# Test on 10 pairs first:
python3 model/scripts/run_teacher.py --pairs_dir data/pairs --max_pairs 10

# Better quality (uses more VRAM, more time):
python3 model/scripts/run_teacher.py \
    --pairs_dir data/pairs \
    --ckpt model/teachers/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
    --inf_h 512 --inf_w 832 --save_h 512 --save_w 832 \
    --valid_iters 24
```

Questions / problems: ping the project owner.
