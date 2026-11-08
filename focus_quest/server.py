"""Focus Quest server: EEG -> engine -> browser UI (websocket) + OpenAI check-ins.

  python server.py --demo        simulated EEG, no hardware, short check-ins (every 40 s)
  python server.py               real NeuroPawnKnight LSL stream, check-ins every 3 minutes
"""
import argparse
import asyncio
import functools
import json
import os
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import websockets

from coach import Coach
from engine import FocusEngine
from sources import DemoSource, LSLSource

HERE = Path(__file__).parent


def load_env():
    p = HERE / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


async def main(a):
    load_env()
    coach = Coach()
    src = DemoSource() if a.demo else LSLSource()
    chunk_s = a.chunk or (40 if a.demo else 180)
    calib_s = a.calib or (10 if a.demo else 20)
    eng = FocusEngine(src.fs, src.n_ch, calib_s=calib_s, factor=a.factor)
    clients = set()
    hello = {"type": "hello", "fs": src.fs, "demo": a.demo, "chunk_s": chunk_s,
             "calib_s": calib_s, "n_ch": src.n_ch, "labels": src.labels,
             "ai": coach.client is not None, "model": coach.model}

    async def handler(ws):
        clients.add(ws)
        await ws.send(json.dumps(hello))
        try:
            async for msg in ws:
                if json.loads(msg).get("cmd") == "calibrate":
                    eng.start_calibration(time.monotonic())
        finally:
            clients.discard(ws)

    def send(obj):
        websockets.broadcast(clients, json.dumps(obj))

    handler_cls = functools.partial(Quiet, directory=str(HERE / "web"))
    httpd = ThreadingHTTPServer(("localhost", a.http_port), handler_cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://localhost:{a.http_port}"
    print(f"Focus Quest running at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)

    loop, loop_t = asyncio.get_running_loop(), time.monotonic()
    t_state, t_chunk, n_chunk, auto = 0.0, loop_t, 0, a.demo
    async with websockets.serve(handler, "localhost", a.port):
        while True:
            await asyncio.sleep(0.1)
            now = time.monotonic()
            data = src.read()
            frame = {"type": "frame", "chunk_left": max(0, chunk_s - (now - t_chunk))}
            if data is not None:
                bands, chans = eng.push(data)
                frame["bands"] = {k: np.round(v, 2).tolist() for k, v in bands.items()}
                frame["ch"] = {k: np.round(v, 1).tolist() for k, v in chans.items()}
            if auto and eng.filled >= eng.N:
                eng.start_calibration(now)
                auto = False
            if now - t_state >= 0.5:
                t_state, frame["state"] = now, eng.update(now)
            send(frame)
            if now - t_chunk >= chunk_s:
                t_chunk, n_chunk = now, n_chunk + 1
                s = eng.pop_chunk()
                if s["steps"] == 0:
                    continue
                drifted = s["distracted_frac"] >= a.trigger
                send({"type": "checkin", "n": n_chunk, "summary": s, "nudge": drifted})
                if drifted:
                    text, how = await loop.run_in_executor(None, coach.nudge, s)
                    send({"type": "coach", "text": text, "ai": how, "n": n_chunk, "summary": s})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="simulated EEG, no hardware")
    ap.add_argument("--chunk", type=int, default=0, help="check-in length in seconds (180)")
    ap.add_argument("--calib", type=int, default=0, help="calibration seconds (20)")
    ap.add_argument("--factor", type=float, default=2.0,
                    help="movement vs calibration that counts as drifting (lower = more sensitive)")
    ap.add_argument("--trigger", type=float, default=0.4,
                    help="share of the check-in spent drifting before Pip says anything")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--http-port", type=int, default=8000)
    try:
        asyncio.run(main(ap.parse_args()))
    except KeyboardInterrupt:
        pass
