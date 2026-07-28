"""Fetch InStereo2K.zip from the OneDrive zip-transform endpoint straight into
the stereo-datasets Modal volume (the download token is embedded and time
limited, so run this promptly).

    modal run model/scripts/modal/fetch_instereo2k.py::fetch
"""
from __future__ import annotations
import modal

app = modal.App("fetch-instereo2k")
vol = modal.Volume.from_name("stereo-datasets")
image = modal.Image.debian_slim(python_version="3.12").apt_install("curl", "unzip")

URL = "https://eastus1-mediap.svc.ms/transform/zip?cs=MDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwfFNQTw"
DATA = ('zipFileName=InStereo2K.zip&guid=13f42bf0-d63c-45f7-a2d8-5088dab1fca1'
        '&provider=spo&files=%7B%22items%22%3A%5B%7B%22name%22%3A%22InStereo2K%22'
        '%2C%22size%22%3A0%2C%22docId%22%3A%22https%3A%2F%2Fonedrive.live.com%3A443'
        '%2F_api%2Fv2.0%2Fdrives%2Fb%21Hpe4mazYIkqkFbwnaEqGhdKsxKC9z6FMvoH42F0ZJIf0'
        'N31S-rLtTZD3CjGAm_kL%2Fitems%2F01VTRDN3MEDRIQUOOYYMQIBOTIAAAAAAAA%3Fversion'
        '%3DPublished%26access_token%3Dv1e.eyJzaXRlaWQiOiI5OWI4OTcxZS1kOGFjLTRhMjIt'
        'YTQxNS1iYzI3Njg0YTg2ODUiLCJhcHBpZCI6IjAwMDAwMDAzLTAwMDAtMGZmMS1jZTAwLTAwMDAw'
        'MDAwMDAwMCIsImF1ZCI6IjAwMDAwMDAzLTAwMDAtMGZmMS1jZTAwLTAwMDAwMDAwMDAwMC9vbmVk'
        'cml2ZS5saXZlLmNvbUA5MTg4MDQwZC02YzY3LTRjNWItYjExMi0zNmEzMDRiNjZkYWQiLCJleHAi'
        'OiIxNzc2NDE2NDAwIn0.EhzqLndO_UfZH235twzknT11X9lnf6xgZ03U60TI2SndZLaq8Wv0npm'
        'RaSmK8xhKuoZEqzH_d1dx0rFQ4e2I0dO7dx7gCrxpz0npeQzUOhbyLed-Qa1fonKiTAJHn2BxtA'
        'PI0leLfJAN25NftCQBfxf4VfyAQoXduI2edvPoUi0P7fOo7miHa89oCUfHtPyfmJA3uroRhUt-a5'
        'jWH8zH1rScnuBIFzR20MkVhG3P1zYqQ3egOyyK9Et0zNDx9rUt3GpjKLaYMxs4FfMX8ffN8X2O_'
        'kPVz1EKcqk8cE03Tg8sGu6ykcrwl8eJPjr3PBDN2BXLpGZ72aWA8qQGu2vZCQ.u235EaUBi3GVt'
        'QPK5zVBkGmUE3qHnjjonSf038WCpMY%22%2C%22isFolder%22%3Atrue%7D%5D%7D'
        '&oAuthToken=&pacToken=')


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def fetch():
    import os
    import subprocess
    os.makedirs("/data/instereo2k", exist_ok=True)
    out = "/data/instereo2k/InStereo2K.zip"
    cmd = ["curl", URL, "-sS", "-L", "--fail",
           "-H", "content-type: application/x-www-form-urlencoded",
           "-H", "origin: https://onedrive.live.com",
           "-H", ("user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"),
           "--data-raw", DATA, "-o", out]
    print("downloading InStereo2K.zip ...", flush=True)
    subprocess.run(cmd, check=True)
    sz = os.path.getsize(out) / 1e6
    print(f"downloaded {sz:.0f} MB -> {out}", flush=True)
    # sanity: is it a real zip?
    r = subprocess.run(["unzip", "-l", out], capture_output=True, text=True)
    print(r.stdout[-800:] if r.returncode == 0 else f"UNZIP CHECK FAILED: {r.stderr[:400]}",
          flush=True)
    vol.commit()
    return sz


@app.local_entrypoint()
def main():
    print("MB:", fetch.remote())
