import modal

app = modal.App("stereo-probe-volume")
vol = modal.Volume.from_name("stereo-probe-data", create_if_missing=True)
img = modal.Image.debian_slim().apt_install("wget")

DEFAULT_URL = "https://download.thinkbroadband.com/50MB.zip"


@app.function(image=img, volumes={"/data": vol}, timeout=600)
def download(url: str):
    import os, subprocess
    os.makedirs("/data", exist_ok=True)
    fname = url.rsplit("/", 1)[-1] or "blob.bin"
    out = f"/data/{fname}"
    subprocess.check_call(["wget", "-c", "-q", "--show-progress", "-O", out, url])
    sz = os.path.getsize(out)
    vol.commit()
    return f"saved {fname}  {sz/1e6:.2f} MB"


@app.function(volumes={"/data": vol})
def ls():
    import os
    vol.reload()
    if not os.path.isdir("/data"):
        return "(no /data)"
    items = []
    for f in sorted(os.listdir("/data")):
        p = f"/data/{f}"
        sz = os.path.getsize(p)
        items.append(f"{f}  {sz/1e6:.2f} MB")
    return "\n".join(items) if items else "(empty)"


@app.local_entrypoint()
def main(action: str = "ls", url: str = DEFAULT_URL):
    if action == "download":
        print(download.remote(url))
    elif action == "ls":
        print(ls.remote())
    else:
        print(f"action must be 'ls' or 'download', got {action!r}")
