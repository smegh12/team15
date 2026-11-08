"""Signal processing and the focus/drift state machine (no I/O here)."""
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
ART_UV = 200.0  # peak-to-peak above this (after detrend) = blink/movement artifact


def detrend(win):
    t = np.linspace(-1, 1, win.shape[1])
    slope = (win * t).sum(axis=1, keepdims=True) / (t * t).sum()
    return win - win.mean(axis=1, keepdims=True) - slope * t


def metrics(win, fs):
    """Relative band powers + log engagement index beta/(alpha+theta); None if too noisy."""
    win = detrend(win)
    p2p = win.max(axis=1) - win.min(axis=1)
    good = (p2p < ART_UV) & (p2p > 1.0)  # too big = artifact, too flat = no contact
    if good.sum() < max(1, win.shape[0] // 2):
        return None
    psd = np.abs(np.fft.rfft(win[good] * np.hanning(win.shape[1]), axis=1)) ** 2
    f = np.fft.rfftfreq(win.shape[1], 1 / fs)
    p = {k: psd[:, (f >= lo) & (f < hi)].sum(axis=1).mean() for k, (lo, hi) in BANDS.items()}
    tot = sum(p.values()) + 1e-12
    eng = float(np.log((p["beta"] + 1e-12) / (p["alpha"] + p["theta"] + 1e-12)))
    return {"logE": eng, **{k: v / tot for k, v in p.items()}}


def display_sos(fs):
    """1-40 Hz band-pass plus a 60 Hz mains notch, for the per-channel view."""
    hi = min(40.0, fs / 2 - 1)
    sos = butter(2, (1.0, hi), btype="band", fs=fs, output="sos")
    if 60 < fs / 2 - 1:
        b, a = iirnotch(60.0, 30.0, fs=fs)
        sos = np.vstack([sos, tf2sos(b, a)])
    return sos


class FocusEngine:
    def __init__(self, fs, n_ch, calib_s=60.0, thr=1.2, hold_s=6.0, recover=0.75, win_s=4.0):
        self.fs, self.calib_s, self.thr, self.hold_s, self.recover = fs, calib_s, thr, hold_s, recover
        self.N = int(fs * win_s)
        self.buf, self.filled = np.zeros((n_ch, self.N)), 0
        self.sos = {b: butter(3, e, btype="band", fs=fs, output="sos") for b, e in BANDS.items()}
        self.dsos = display_sos(fs)
        self.zi, self.dzi = None, None
        self.mode, self.calib, self.t_cal = "idle", [], 0.0
        self.mu, self.sigma = 0.0, 1.0
        self.ema, self.low_since, self.distracted = None, None, False
        self.calib_failed = False
        self.reset_chunk()

    def reset_chunk(self):
        self.c = {"steps": 0, "dist": 0, "art": 0, "z": []}

    def start_calibration(self, now):
        self.mode, self.calib, self.t_cal = "calibrating", [], now
        self.ema, self.low_since, self.distracted = None, None, False
        self.calib_failed = False
        self.reset_chunk()

    def push(self, data):
        """data: (n_ch, k) new samples -> (band traces of channel average, per-channel traces)."""
        k = min(data.shape[1], self.N)
        self.buf = np.roll(self.buf, -k, axis=1)
        self.buf[:, -k:] = data[:, -k:]
        self.filled += k
        mean = data.mean(axis=0)
        if self.zi is None:  # start filters at the signal's offset to avoid a startup spike
            self.zi = {b: sosfilt_zi(s) * mean[0] for b, s in self.sos.items()}
            self.dzi = sosfilt_zi(self.dsos)[:, None, :] * data[:, :1][None, :, :]
        bands = {}
        for b, sos in self.sos.items():
            bands[b], self.zi[b] = sosfilt(sos, mean, zi=self.zi[b])
        clean, self.dzi = sosfilt(self.dsos, data, axis=1, zi=self.dzi)
        n = min(self.filled, self.N)
        raw = data - self.buf[:, -n:].mean(axis=1, keepdims=True)
        return bands, {"clean": clean, "raw": raw}

    def update(self, now):
        s = {"state": "idle", "score": None, "theta": 0.0, "alpha": 0.0, "beta": 0.0,
             "left": 0.0, "calib_failed": self.calib_failed}
        if self.filled < self.N:
            s["state"] = "filling"
            return s
        m = metrics(self.buf, self.fs)
        monitoring = self.mode == "monitoring"
        if monitoring:
            self.c["steps"] += 1
        if m is None:
            s["state"] = "artifact"
            if monitoring:
                self.c["art"] += 1
            return s
        s.update({k: round(m[k], 3) for k in BANDS})
        if self.mode == "calibrating":
            self.calib.append(m["logE"])
            s["state"], s["left"] = "calibrating", max(0.0, self.calib_s - (now - self.t_cal))
            if s["left"] <= 0:
                self._finish()
        elif monitoring:
            z = (m["logE"] - self.mu) / self.sigma
            self.ema = z if self.ema is None else 0.5 * self.ema + 0.5 * z
            if self.ema < -self.thr:
                self.low_since = self.low_since or now
                if now - self.low_since >= self.hold_s:  # must stay low the whole time
                    self.distracted = True
            else:
                self.low_since = None
                if self.ema > -self.thr * self.recover:  # snap back to focused quickly
                    self.distracted = False
            s["score"] = float(np.clip(self.ema, -4, 4))
            s["state"] = "distracted" if self.distracted else "focused"
            self.c["z"].append(self.ema)
            self.c["dist"] += self.distracted
        return s

    def _finish(self):
        v = np.array(self.calib)
        if len(v) < 10:
            self.mode, self.calib_failed = "idle", True
            return
        self.mu, self.sigma, self.mode = float(v.mean()), max(float(v.std()), 0.1), "monitoring"
        self.reset_chunk()

    def pop_chunk(self):
        c = self.c
        self.reset_chunk()
        clean = max(c["steps"] - c["art"], 1)
        return {"steps": c["steps"], "distracted_frac": round(c["dist"] / clean, 2),
                "artifact_frac": round(c["art"] / max(c["steps"], 1), 2),
                "mean_z": round(float(np.mean(c["z"])), 2) if c["z"] else 0.0}
