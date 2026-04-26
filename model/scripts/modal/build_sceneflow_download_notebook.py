"""Build sceneflow_download_modal.ipynb — a Jupyter notebook for Modal.com.

Downloads the full Scene Flow corpus (Driving + Monkaa + FlyingThings3D,
finalpass RGB + dense disparity, ~205 GB total) into a Modal Volume so
the data persists across notebook sessions and can be mounted into
training jobs later.

Each tarball gets its own download cell + verification cell, so any
single failure can be retried in isolation. wget -c resumes partial
downloads on rerun. File sizes are sanity-checked against published
expected sizes; mismatches print a loud warning.

Run:
    python3 model/scripts/modal/build_sceneflow_download_notebook.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent / "sceneflow_download_modal.ipynb"


# Scene Flow file table — all from Freiburg LMB.
BASE = ("https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/"
        "Release_april16/data")

FILES = [
    # (subset, kind, url, expected_size_bytes_min, expected_size_bytes_max)
    ("Driving",         "frames_finalpass",
     f"{BASE}/Driving/raw_data/driving__frames_finalpass.tar",
     6_500_000_000, 7_300_000_000),
    ("Driving",         "disparity",
     f"{BASE}/Driving/derived_data/driving__disparity.tar.bz2",
     9_000_000_000, 10_500_000_000),
    ("Monkaa",          "frames_finalpass",
     f"{BASE}/Monkaa/raw_data/monkaa__frames_finalpass.tar",
     35_000_000_000, 42_000_000_000),
    ("Monkaa",          "disparity",
     f"{BASE}/Monkaa/derived_data/monkaa__disparity.tar.bz2",
     11_000_000_000, 16_000_000_000),
    ("FlyingThings3D",  "frames_finalpass",
     f"{BASE}/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar",
     45_000_000_000, 50_000_000_000),
    ("FlyingThings3D",  "disparity",
     f"{BASE}/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2",
     90_000_000_000, 100_000_000_000),
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


cells: list[dict] = []

# ----------------------------- Header ----------------------------------- #
cells.append(md(
    "# Scene Flow full-corpus downloader (Modal-friendly)\n"
    "\n"
    "Downloads the **full Scene Flow** dataset (Driving + Monkaa + "
    "FlyingThings3D, finalpass RGB + dense disparity, **~205 GB total**) "
    "from the official Freiburg LMB server into a persistent location.\n"
    "\n"
    "**Designed for Modal.com**: when run on a Modal notebook with a Volume "
    "mounted at `/data` (or wherever you choose), the downloaded files "
    "persist across sessions and can be mounted into training jobs later. "
    "The notebook also runs unchanged on a plain Jupyter / Colab / local "
    "machine — just set `TARGET` to whatever path you want.\n"
    "\n"
    "## Safety design\n"
    "\n"
    "- **One cell per file.** If a download fails or stalls, you can re-run "
    "  just that one cell.\n"
    "- **Resume on re-run.** Every download uses `wget -c` (continue), so "
    "  re-running a cell after an interruption picks up from the last byte.\n"
    "- **Size verification.** After each download, the file size is checked "
    "  against published expected ranges and prints a loud warning on "
    "  mismatch.\n"
    "- **No early bail.** A failed download in one cell does not prevent "
    "  others from proceeding.\n"
    "- **No deletion.** Downloaded tarballs are never auto-removed; you "
    "  decide when to extract or discard them.\n"
    "\n"
    "## Files downloaded\n"
    "\n"
    "| subset | file | approx. size |\n"
    "|---|---|---|\n"
    "| Driving | `driving__frames_finalpass.tar` | 6.5 GB |\n"
    "| Driving | `driving__disparity.tar.bz2` | 9.5 GB |\n"
    "| Monkaa | `monkaa__frames_finalpass.tar` | 38 GB |\n"
    "| Monkaa | `monkaa__disparity.tar.bz2` | 13 GB |\n"
    "| FlyingThings3D | `flyingthings3d__frames_finalpass.tar` | 45 GB |\n"
    "| FlyingThings3D | `flyingthings3d__disparity.tar.bz2` | 93 GB |\n"
    "| **total** | | **~205 GB** |\n"
    "\n"
    "Freiburg's outbound bandwidth varies; expect 1–10 MB/s per stream. "
    "Allow several hours for the whole corpus (FlyingThings3D disparity "
    "alone can take 6+ hours on a slow connection).\n"
))


# ----------------------------- Cell 1: Setup ---------------------------- #
cells.append(md("## 1. Setup — mount target directory and verify space"))
cells.append(code(
    "import os, shutil, subprocess\n"
    "from pathlib import Path\n"
    "\n"
    "# ---- TARGET DIRECTORY ----\n"
    "# On Modal: set this to your mounted Volume path (e.g. '/data/sceneflow').\n"
    "# On other Jupyter envs: any writable path with ~210 GB free.\n"
    "TARGET = Path('/data/sceneflow')\n"
    "\n"
    "TARGET.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "# Disk-space sanity check\n"
    "stat = shutil.disk_usage(TARGET)\n"
    "free_gb = stat.free / 1e9\n"
    "needed_gb = 210\n"
    "print(f'TARGET = {TARGET}')\n"
    "print(f'free disk space at TARGET: {free_gb:.1f} GB')\n"
    "print(f'estimated required: {needed_gb} GB (downloads only; +200 GB for "
    "extracted)')\n"
    "if free_gb < needed_gb:\n"
    "    print(f'\\n!! WARNING: only {free_gb:.0f} GB free; downloading the '\n"
    "          f'full corpus needs ~{needed_gb} GB. Make sure your Modal '\n"
    "          f'Volume is large enough or change TARGET.')\n"
    "\n"
    "# Verify wget is installed (it should be on every Linux container)\n"
    "rc = subprocess.call(['which', 'wget'], stdout=subprocess.DEVNULL)\n"
    "if rc != 0:\n"
    "    print('!! wget missing — install it now:')\n"
    "    !apt-get -qq install -y wget\n"
))


# ----------------------------- Cell 2: Helpers -------------------------- #
cells.append(md("## 2. Helper functions"))
cells.append(code(
    "import time, subprocess\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def fmt_bytes(b):\n"
    "    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):\n"
    "        if b < 1024:\n"
    "            return f'{b:.2f} {unit}'\n"
    "        b /= 1024\n"
    "    return f'{b:.2f} PB'\n"
    "\n"
    "\n"
    "def expected_status(path: Path, lo: int, hi: int) -> str:\n"
    "    if not path.exists():\n"
    "        return 'missing'\n"
    "    sz = path.stat().st_size\n"
    "    if sz < lo:\n"
    "        return f'partial ({fmt_bytes(sz)} of expected ~{fmt_bytes(lo)})'\n"
    "    if sz > hi * 1.1:  # generous upper-bound fudge\n"
    "        return f'oversize ({fmt_bytes(sz)})'\n"
    "    return f'OK ({fmt_bytes(sz)})'\n"
    "\n"
    "\n"
    "def download_one(url: str, target_dir: Path, lo_bytes: int, hi_bytes: int,\n"
    "                  max_attempts: int = 50, retry_sleep: int = 60) -> bool:\n"
    "    fname = url.rsplit('/', 1)[-1]\n"
    "    out = target_dir / fname\n"
    "    print(f'\\n=== {fname} ===')\n"
    "    print(f'url:    {url}')\n"
    "    print(f'target: {out}')\n"
    "    status = expected_status(out, lo_bytes, hi_bytes)\n"
    "    print(f'status before: {status}')\n"
    "    if status.startswith('OK'):\n"
    "        print('already complete — skipping.')\n"
    "        return True\n"
    "    attempt = 1\n"
    "    while attempt <= max_attempts:\n"
    "        print(f'\\n--- attempt {attempt}/{max_attempts}, '\n"
    "              f'started {time.strftime(\"%H:%M:%S\")} ---')\n"
    "        cmd = [\n"
    "            'wget', '-c',\n"
    "            '--tries=0',\n"
    "            '--waitretry=30',\n"
    "            '--timeout=120',\n"
    "            '--read-timeout=300',\n"
    "            '--retry-connrefused',\n"
    "            '--show-progress',\n"
    "            '--progress=bar:force:noscroll',\n"
    "            '-O', str(out), url,\n"
    "        ]\n"
    "        rc = subprocess.call(cmd)\n"
    "        status = expected_status(out, lo_bytes, hi_bytes)\n"
    "        print(f'status after attempt: {status}  (wget rc={rc})')\n"
    "        if status.startswith('OK'):\n"
    "            print(f'!! download complete: {fname}')\n"
    "            return True\n"
    "        attempt += 1\n"
    "        if attempt <= max_attempts:\n"
    "            print(f'sleeping {retry_sleep}s before retry...')\n"
    "            time.sleep(retry_sleep)\n"
    "    print(f'!! gave up on {fname} after {max_attempts} attempts.')\n"
    "    return False\n"
    "\n"
    "print('helpers ready.')\n"
))


# ----------------------------- Cell 3: Connectivity --------------------- #
cells.append(md(
    "## 3. Connectivity & permission check\n"
    "\n"
    "Quick HEAD request against Freiburg before committing to a 200 GB "
    "download. If this fails, your environment cannot reach the data and "
    "no further cell will succeed."
))
cells.append(code(
    "test_url = (\n"
    "    'https://lmb.informatik.uni-freiburg.de/data/'\n"
    "    'SceneFlowDatasets_CVPR16/Release_april16/data/'\n"
    "    'Driving/raw_data/driving__frames_finalpass.tar'\n"
    ")\n"
    "import subprocess\n"
    "out = subprocess.run(\n"
    "    ['wget', '--spider', '--server-response', '--max-redirect=2',\n"
    "     '--timeout=20', '-q', '-O', '-', test_url],\n"
    "    capture_output=True, text=True)\n"
    "print('rc:', out.returncode)\n"
    "print('--- server response ---')\n"
    "print(out.stderr[-1500:])\n"
    "if out.returncode == 0:\n"
    "    print('\\n[OK] connectivity to Freiburg looks good. Proceed.')\n"
    "else:\n"
    "    print('\\n[FAIL] cannot reach Freiburg. Check Modal egress / DNS / '\n"
    "          'IP allowlist before starting downloads.')\n"
))


# ----------------------------- Per-file cells --------------------------- #
for i, (subset, kind, url, lo, hi) in enumerate(FILES, start=4):
    fname = url.rsplit("/", 1)[-1]
    short = f"{subset}: {kind}"
    cells.append(md(
        f"## {i}. Download `{fname}`\n"
        f"\n"
        f"**Subset:** {subset}  \n"
        f"**Kind:** {kind}  \n"
        f"**Approx. size:** {lo // 1_000_000_000}--{hi // 1_000_000_000} GB  \n"
        f"\n"
        f"Re-running this cell after a partial download will resume from the "
        f"last byte (wget `-c`)."
    ))
    cells.append(code(
        f"download_one(\n"
        f"    url={url!r},\n"
        f"    target_dir=TARGET,\n"
        f"    lo_bytes={lo},\n"
        f"    hi_bytes={hi},\n"
        f"    max_attempts=50,\n"
        f"    retry_sleep=60,\n"
        f")\n"
    ))


# ----------------------------- Final summary ---------------------------- #
n_summary = len(FILES) + 4
cells.append(md(
    f"## {n_summary}. Final summary\n"
    f"\n"
    f"Re-checks every file, prints sizes, and reports total disk used."
))

summary_files_lines = []
for subset, kind, url, lo, hi in FILES:
    fname = url.rsplit('/', 1)[-1]
    summary_files_lines.append(f"    ({fname!r}, {lo}, {hi}),")
summary_files_block = "\n".join(summary_files_lines)
cells.append(code(
    f"FILES = [\n"
    f"{summary_files_block}\n"
    f"]\n"
    f"\n"
    f"total_bytes = 0\n"
    f"all_ok = True\n"
    f"print(f'{{\"file\":<55s}} {{\"status\":<35s}} {{\"size\":>15s}}')\n"
    f"print('-' * 110)\n"
    f"for fname, lo, hi in FILES:\n"
    f"    p = TARGET / fname\n"
    f"    st = expected_status(p, lo, hi)\n"
    f"    sz = p.stat().st_size if p.exists() else 0\n"
    f"    total_bytes += sz\n"
    f"    if not st.startswith('OK'):\n"
    f"        all_ok = False\n"
    f"    print(f'{{fname:<55s}} {{st:<35s}} {{fmt_bytes(sz):>15s}}')\n"
    f"print('-' * 110)\n"
    f"print(f'TOTAL: {{fmt_bytes(total_bytes)}}')\n"
    f"print()\n"
    f"if all_ok:\n"
    f"    print('[OK] all 6 tarballs present and within expected size ranges.')\n"
    f"else:\n"
    f"    print('[INCOMPLETE] some files are missing or partial. Re-run the '\n"
    f"          'cells flagged above.')\n"
))


# ----------------------------- Optional: extract ------------------------ #
n_extract = n_summary + 1
cells.append(md(
    f"## {n_extract}. (Optional) Extract\n"
    f"\n"
    f"Extracting the full corpus consumes about another **200 GB** on top of "
    f"the tarballs. Don't run this unless you have space — many users keep "
    f"only the tars in the Modal Volume and extract on-demand inside their "
    f"training job. Each tar/bz2 has its own cell so partial extraction is "
    f"fine.\n"
    f"\n"
    f"**Warning:** the FlyingThings3D disparity bz2 is the slowest to "
    f"decompress (it can take 30+ minutes single-threaded)."
))
cells.append(code(
    "import subprocess, shutil\n"
    "from pathlib import Path\n"
    "\n"
    "stat = shutil.disk_usage(TARGET)\n"
    "free_gb = stat.free / 1e9\n"
    "print(f'free disk before extract: {free_gb:.1f} GB')\n"
    "if free_gb < 220:\n"
    "    raise RuntimeError(\n"
    "        f'only {free_gb:.0f} GB free — extracting the full corpus needs '\n"
    "        f'~200 GB more. Free up space or skip this cell.')\n"
    "\n"
    "for tar_name in [\n"
    "    'driving__frames_finalpass.tar',\n"
    "    'driving__disparity.tar.bz2',\n"
    "    'monkaa__frames_finalpass.tar',\n"
    "    'monkaa__disparity.tar.bz2',\n"
    "    'flyingthings3d__frames_finalpass.tar',\n"
    "    'flyingthings3d__disparity.tar.bz2',\n"
    "]:\n"
    "    tar_path = TARGET / tar_name\n"
    "    if not tar_path.exists():\n"
    "        print(f'[SKIP] {tar_name} not present')\n"
    "        continue\n"
    "    print(f'\\n=== extracting {tar_name} ===')\n"
    "    flag = 'xjf' if tar_name.endswith('.bz2') else 'xf'\n"
    "    rc = subprocess.call(['tar', flag, str(tar_path), '-C', str(TARGET)])\n"
    "    print(f'rc={rc}  (0 = ok)')\n"
    "\n"
    "print('\\n--- top-level tree under TARGET ---')\n"
    "for p in sorted(TARGET.iterdir()):\n"
    "    if p.is_dir():\n"
    "        print(f'  {p.name}/')\n"
))


# ----------------------------- Closing -------------------------------- #
cells.append(md(
    "---\n\n"
    "### Notes\n\n"
    "- **Modal Volume mount**: configure your Modal Notebook with "
    "`modal.Volume.from_name('sceneflow', create_if_missing=True)` mounted "
    "at `/data/sceneflow`. The volume persists across sessions; downloads "
    "survive notebook shutdown.\n\n"
    "- **Auto-shutdown**: set the notebook's idle timeout to at least 8 hours "
    "if you plan to leave the FlyingThings3D download running unattended.\n\n"
    "- **Resuming**: any cell can be re-run safely. If a download was 80% "
    "complete when the kernel died, re-running its cell will continue from "
    "byte 80%, not restart from zero.\n\n"
    "- **After completion**: in your downstream training job (also on Modal "
    "or wherever), mount the same Volume read-only and point your data "
    "loader at the extracted directories.\n\n"
    "- **What's NOT included**: optical flow, disparity_change, cleanpass "
    "RGB, motion/depth-boundary weights, occlusion masks. Add cells for "
    "those if you need them — same URL pattern, just swap the tar name.\n"
))


# ----------------------------- Write notebook -------------------------- #
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
