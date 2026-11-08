"""EEG sources: the real LSL stream, or a simulated one for demos."""
import time

import numpy as np


class LSLSource:
    def __init__(self, name="NeuroPawnKnight"):
        from pylsl import StreamInlet, resolve_streams
        found = [s for s in resolve_streams(wait_time=3.0) if s.name() == name]
        if not found:
            raise SystemExit(f"No '{name}' LSL stream found. Start knight_lsl_gui.py first "
                             "(or run this with --demo).")
        self.inlet = StreamInlet(found[0], max_buflen=10)
        info = self.inlet.info()
        n, self.fs = info.channel_count(), info.nominal_srate() or 125.0
        labels = []
        try:
            ch = info.desc().child("channels").child("channel")
            for _ in range(n):
                labels.append(ch.child_value("label") or "?")
                ch = ch.next_sibling()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        self.idx = [i for i, lab in enumerate(labels) if lab.startswith("EXG")][:8] \
            or list(range(min(n, 8)))
        self.n_ch = len(self.idx)
        self.labels = [f"CH{i + 1}" for i in range(self.n_ch)]
        print(f"[lsl] {n} channels at {self.fs:g} Hz, using {self.n_ch}: "
              f"{[labels[i] if i < len(labels) else i for i in self.idx]}")

    def read(self):
        chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=1024)
        return np.asarray(chunk, dtype=float).T[self.idx] if chunk else None


class DemoSource:
    """Fake EEG: 40 s 'focused' (strong beta) then 40 s 'drifting' (strong alpha/theta)."""
    fs, n_ch = 125.0, 8
    labels = [f"CH{i + 1}" for i in range(8)]

    def __init__(self):
        self.t0, self.n, self.rng = time.monotonic(), 0, np.random.default_rng(1)
        self.gain = 0.7 + 0.6 * self.rng.random(self.n_ch)

    def read(self):
        target = int((time.monotonic() - self.t0) * self.fs)
        k = target - self.n
        if k <= 0:
            return None
        t = (self.n + np.arange(k)) / self.fs
        self.n = target
        focus = (t // 40) % 2 == 0
        a, th, b = (np.where(focus, x, y) for x, y in ((6, 22), (5, 16), (12, 4)))
        w = 2 * np.pi
        return np.array([self.gain[c] * (a * np.sin(w * 10 * t + c * .7)
                                         + th * np.sin(w * 6 * t + c)
                                         + b * np.sin(w * 20 * t + c * .5))
                         + self.rng.normal(0, 3, k) for c in range(self.n_ch)])
