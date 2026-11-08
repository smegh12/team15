"""Focus vs drift from eye/head movement only.

Looking around shows up in EEG as large, slow deflections (0.5-10 Hz). We measure
how big those are on each channel, compare to a calibration baseline recorded while
the user works normally, and call it DRIFTING when movement stays well above it."""
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}  # display only


def display_sos(fs):
    """1-40 Hz band-pass plus a 60 Hz mains notch, for the per-channel view."""
    hi = min(40.0, fs / 2 - 1)
    sos = butter(2, (1.0, hi), btype="band", fs=fs, output="sos")
    if 60 < fs / 2 - 1:
        b, a = iirnotch(60.0, 30.0, fs=fs)
        sos = np.vstack([sos, tf2sos(b, a)])
    return sos


class FocusEngine:
    def __init__(self, fs, n_ch, calib_s=20.0, factor=2.0, on_s=1.0, off_s=2.0, win_s=2.0):
        self.fs, self.calib_s, self.factor = fs, calib_s, factor
        self.on_s, self.off_s = on_s, off_s
        self.N = int(fs * win_s)
        self.buf, self.filled = np.zeros((n_ch, self.N)), 0
        self.sos = {b: butter(3, e, btype="band", fs=fs, output="sos") for b, e in BANDS.items()}
        self.dsos = display_sos(fs)
        self.esos = butter(2, (0.5, 10.0), btype="band", fs=fs, output="sos")  # eye/head band
        self.zi = self.dzi = self.ezi = self.dc = None
        self.mode, self.calib, self.t_cal, self.base = "idle", [], 0.0, None
        self.drifting, self.move_since, self.calm_since = False, None, None
        self.calib_failed = False
        self.reset_chunk()

    def reset_chunk(self):
        self.c = {"steps": 0, "dist": 0}

    def start_calibration(self, now):
        self.mode, self.calib, self.t_cal = "calibrating", [], now
        self.drifting, self.move_since, self.calm_since = False, None, None
        self.calib_failed = False
        self.reset_chunk()

    def push(self, data):
        """data: (n_ch, k) new samples -> (band traces of channel average, per-channel traces)."""
        if self.zi is None:  # start filters at the signal's offset to avoid a startup spike
            m0 = data.mean(axis=0)[0]
            self.zi = {b: sosfilt_zi(s) * m0 for b, s in self.sos.items()}
            def init(sos):
                return sosfilt_zi(sos)[:, None, :] * data[:, :1][None, :, :]
            self.dzi, self.ezi = init(self.dsos), init(self.esos)
            self.dc = data[:, 0].astype(float).copy()
        mean = data.mean(axis=0)
        bands = {}
        for b, sos in self.sos.items():
            bands[b], self.zi[b] = sosfilt(sos, mean, zi=self.zi[b])
        clean, self.dzi = sosfilt(self.dsos, data, axis=1, zi=self.dzi)
        eye, self.ezi = sosfilt(self.esos, data, axis=1, zi=self.ezi)
        k = min(data.shape[1], self.N)
        self.buf = np.roll(self.buf, -k, axis=1)
        self.buf[:, -k:] = eye[:, -k:]
        self.filled += k
        self.dc = 0.9 * self.dc + 0.1 * data.mean(axis=1)
        return bands, {"clean": clean, "raw": data - self.dc[:, None]}

    def update(self, now):
        s = {"state": "idle", "score": None, "left": 0.0, "calib_failed": self.calib_failed}
        if self.filled < self.N:
            s["state"] = "filling"
            return s
        amp = self.buf.std(axis=1)  # movement level per channel (uV)
        if self.mode == "calibrating":
            self.calib.append(amp)
            s["state"], s["left"] = "calibrating", max(0.0, self.calib_s - (now - self.t_cal))
            if s["left"] <= 0:
                self._finish()
            return s
        if self.mode != "monitoring":
            return s
        ratio = float(np.sort(amp / self.base)[-3:].mean())  # 3 most-moving channels
        if ratio > self.factor:  # moving a lot
            self.calm_since = None
            self.move_since = self.move_since or now
            if now - self.move_since >= self.on_s:
                self.drifting = True
        elif ratio < self.factor * 0.8:  # clearly still
            self.move_since = None
            self.calm_since = self.calm_since or now
            if now - self.calm_since >= self.off_s:
                self.drifting = False
        else:  # in between: keep the current state
            self.move_since = self.calm_since = None
        focus = 1 - (ratio - 1) / (2 * max(self.factor - 1, 0.1))
        s["score"] = round(float(np.clip(focus, 0, 1)), 3)
        s["movement"] = round(ratio, 2)
        s["state"] = "distracted" if self.drifting else "focused"
        self.c["steps"] += 1
        self.c["dist"] += self.drifting
        return s

    def _finish(self):
        if len(self.calib) < 5:
            self.mode, self.calib_failed = "idle", True
            return
        self.base = np.maximum(np.median(np.array(self.calib), axis=0), 0.5)
        self.mode = "monitoring"
        self.reset_chunk()

    def pop_chunk(self):
        c = self.c
        self.reset_chunk()
        return {"steps": c["steps"], "distracted_frac": round(c["dist"] / max(c["steps"], 1), 2)}
