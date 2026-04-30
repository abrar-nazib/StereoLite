import modal

app = modal.App("stereo-probe-gpu")


@app.function()
def cpu():
    import shutil, subprocess
    has_smi = shutil.which("nvidia-smi") is not None
    if not has_smi:
        return {"gpu_visible": False, "note": "no nvidia-smi on this container"}
    p = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    return {"gpu_visible": p.returncode == 0, "head": p.stdout[:200]}


@app.function(gpu="T4")
def t4():
    import subprocess
    p = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return {"rc": p.returncode, "info": p.stdout.strip()}


@app.local_entrypoint()
def main(hw: str = "cpu"):
    if hw == "cpu":
        print(cpu.remote())
    elif hw == "t4":
        print(t4.remote())
    else:
        print(f"hw must be 'cpu' or 't4', got {hw!r}")
