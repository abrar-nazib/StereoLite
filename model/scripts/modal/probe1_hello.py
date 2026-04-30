import modal

app = modal.App("stereo-probe-hello")


@app.function()
def hello():
    import platform, socket, time
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "system": platform.system(),
        "epoch": time.time(),
    }


@app.local_entrypoint()
def main():
    print(hello.remote())
