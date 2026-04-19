"""Distill FoundationStereo pseudo-GT into the d1/d2/d3 lightweight students.

Each student is trained independently. The training data is the saved
(left, right, disp_pseudo) triples produced by run_teacher.py. Loss is L1
on disparity with a `valid` mask wherever pseudo-GT is positive and below
a max-disparity threshold.

Output:
    model/checkpoints/student_{d1,d2,d3}.pth
    model/checkpoints/train_{d1,d2,d3}.csv   (step, loss, train_epe)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJ, "model"))
sys.path.insert(0, os.path.join(PROJ, "model", "designs"))


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, ids: list[str], inf_h: int, inf_w: int):
        self.root = Path(root)
        self.ids = ids
        self.h = inf_h
        self.w = inf_w

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        l = cv2.imread(str(self.root / "left" / f"{i}.png"))
        r = cv2.imread(str(self.root / "right" / f"{i}.png"))
        d = np.load(self.root / "disp_pseudo" / f"{i}.npy")

        l = cv2.resize(l, (self.w, self.h))
        r = cv2.resize(r, (self.w, self.h))
        if d.shape != (self.h, self.w):
            sx = self.w / d.shape[1]
            d = cv2.resize(d, (self.w, self.h), interpolation=cv2.INTER_LINEAR) * sx

        l = cv2.cvtColor(l, cv2.COLOR_BGR2RGB).astype(np.float32)
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32)
        return (
            torch.from_numpy(l).permute(2, 0, 1).contiguous(),
            torch.from_numpy(r).permute(2, 0, 1).contiguous(),
            torch.from_numpy(d).unsqueeze(0).contiguous(),
        )


def build_student(name: str):
    if name == "d1":
        from d1_tile import StereoLite
        return StereoLite()
    if name == "d2":
        from d2_cascade import CascadeStereo
        return CascadeStereo()
    if name == "d3":
        from d3_sgm import LearnedSGMStereo
        return LearnedSGMStereo()
    raise ValueError(name)


def loss_l1_masked(pred: torch.Tensor, target: torch.Tensor,
                   max_disp: float = 192.0) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (target > 0.0) & (target < max_disp) & torch.isfinite(target)
    err = (pred - target).abs() * valid
    n = valid.sum().clamp(min=1.0)
    return err.sum() / n, err.sum() / n   # (loss, epe)


def split_ids(root: str, val_count: int) -> tuple[list[str], list[str]]:
    ids = sorted([f.replace(".npy", "") for f in os.listdir(os.path.join(root, "disp_pseudo"))])
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    return ids[val_count:], ids[:val_count]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True, choices=["d1", "d2", "d3"])
    p.add_argument("--data_root", default=os.path.join(PROJ, "data", "pairs"))
    p.add_argument("--inf_h", type=int, default=384)
    p.add_argument("--inf_w", type=int, default=640)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val_count", type=int, default=30)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--out_dir", default=os.path.join(PROJ, "model", "checkpoints"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device.type}; student={args.student}")

    train_ids, val_ids = split_ids(args.data_root, args.val_count)
    print(f"train pairs={len(train_ids)} val pairs={len(val_ids)}")

    ds = PairDataset(args.data_root, train_ids, args.inf_h, args.inf_w)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch,
                                         shuffle=True, num_workers=2,
                                         persistent_workers=True, pin_memory=True)
    iter_loader = iter(loader)

    model = build_student(args.student).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"student params={n_params/1e6:.3f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                        eta_min=args.lr / 10)

    log_path = os.path.join(args.out_dir, f"train_{args.student}.csv")
    fp_log = open(log_path, "w", newline="")
    w = csv.writer(fp_log)
    w.writerow(["step", "loss", "epe", "lr", "ms_per_step"])

    model.train()
    t0 = time.time()
    running = []
    for step in range(1, args.steps + 1):
        try:
            L, R, D = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            L, R, D = next(iter_loader)
        L, R, D = L.to(device), R.to(device), D.to(device)
        pred = model(L, R)
        if pred.shape != D.shape:
            # in case shapes differ slightly (rounding), align
            pred = torch.nn.functional.interpolate(pred, size=D.shape[-2:],
                                                    mode="bilinear", align_corners=True)
        loss, epe = loss_l1_masked(pred, D)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            ms_per_step = (time.time() - t0) * 1000 / step
            cur_lr = opt.param_groups[0]["lr"]
            mean_loss = float(np.mean(running[-args.log_every:]))
            w.writerow([step, mean_loss, float(epe.item()), cur_lr, ms_per_step])
            fp_log.flush()
            print(f"  step {step:5d}  loss={mean_loss:6.3f}  epe={epe.item():6.3f}  "
                  f"lr={cur_lr:.2e}  {ms_per_step:6.1f} ms/step")

    fp_log.close()
    ckpt_path = os.path.join(args.out_dir, f"student_{args.student}.pth")
    torch.save({"model": model.state_dict(), "params_M": n_params / 1e6,
                "args": vars(args)}, ckpt_path)
    print(f"wrote {ckpt_path}")
    print(f"wrote {log_path}")

    # Quick val EPE
    model.eval()
    val_ds = PairDataset(args.data_root, val_ids, args.inf_h, args.inf_w)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
    epes = []
    with torch.no_grad():
        for L, R, D in val_loader:
            L, R, D = L.to(device), R.to(device), D.to(device)
            pred = model(L, R)
            if pred.shape != D.shape:
                pred = torch.nn.functional.interpolate(pred, size=D.shape[-2:],
                                                        mode="bilinear", align_corners=True)
            _, epe = loss_l1_masked(pred, D)
            epes.append(float(epe.item()))
    val_epe = float(np.mean(epes))
    print(f"VAL EPE vs teacher: {val_epe:.3f} px (n={len(epes)})")
    with open(os.path.join(args.out_dir, f"val_{args.student}.txt"), "w") as fp:
        fp.write(f"val_epe_vs_teacher = {val_epe:.4f}\n"
                 f"n_pairs = {len(epes)}\n"
                 f"params_M = {n_params/1e6:.3f}\n")


if __name__ == "__main__":
    main()
