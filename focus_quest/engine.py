"""Focus vs drift from eye/head movement, calibrated with a dot-tracking game.

Eye and head movements show up in EEG as large, slow deflections (0.5-10 Hz). During
calibration we measure them per channel twice: while the user tracks a dot across the
whole screen (big movements) and while they read a short passage (small movements).
The drift threshold sits between the two, so reading counts as focused and big looks
around count as drifting."""
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
    def __init__(self, fs, n_ch, factor=2.0, on_s=1.0, off_s=2.0, win_s=2.0):
        self.fs, self.factor, self.on_s, self.off_s, self.win_s = fs, factor, on_s, off_s, win_s
        self.N = int(fs * win_s)
        self.buf, self.filled = np.zeros((n_ch, self.N)), 0
        self.sos = {b: butter(3, e, btype="band", fs=fs, output="sos") for b, e in BANDS.items()}
        self.dsos = display_sos(fs)
        self.esos = butter(2, (0.5, 10.0), btype="band", fs=fs, output="sos")  # eye/head band
        self.zi = self.dzi = self.ezi = self.dc = None
        self.mode, self.phase, self.phase_t = "idle", None, 0.0
        self.samples = {"track": [], "read": []}
        self.thr, self.calib_info, self.calib_failed = None, None, False
        self.drifting, self.move_since, self.calm_since = False, None, None
        self.reset_chunk()

    def reset_chunk(self):
        self.c = {"steps": 0, "dist": 0}

    def calib_phase(self, phase, now):
        """Called by the calibration game: 'track', 'read', 'done' or 'cancel'."""
        if phase == "track":
            self.samples = {"track": [], "read": []}
            self.calib_failed = False
        if phase in ("track", "read"):
            self.mode, self.phase, self.phase_t = "calibrating", phase, now
            self.drifting, self.move_since, self.calm_since = False, None, None
        elif phase == "done":
            self._finish()
        elif phase == "cancel":
            self.mode = "monitoring" if self.thr is not None else "idle"
            self.phase = None

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
        s = {"state": "idle", "score": None}
        if self.filled < self.N:
            s["state"] = "filling"
            return s
        amp = self.buf.std(axis=1)  # movement level per channel (uV)
        if self.mode == "calibrating":
            if self.phase in self.samples and now - self.phase_t >= self.win_s:
                self.samples[self.phase].append(amp)
            s["state"] = "calibrating"
            return s
        if self.mode != "monitoring":
            return s
        ratio = float(np.sort(amp / self.thr)[-3:].mean())  # 3 most-moving channels
        if ratio > 1.0:  # bigger than the reading/tracking midpoint
            self.calm_since = None
            self.move_since = self.move_since or now
            if now - self.move_since >= self.on_s:
                self.drifting = True
        elif ratio < 0.8:  # clearly reading-sized or still
            self.move_since = None
            self.calm_since = self.calm_since or now
            if now - self.calm_since >= self.off_s:
                self.drifting = False
        else:
            self.move_since = self.calm_since = None
        s["score"] = round(float(np.clip(1 - 0.5 * ratio, 0, 1)), 3)
        s["movement"] = round(ratio, 2)
        s["state"] = "distracted" if self.drifting else "focused"
        self.c["steps"] += 1
        self.c["dist"] += self.drifting
        return s

    def _finish(self):
        tr, rd = self.samples["track"], self.samples["read"]
        self.phase = None
        if len(tr) < 4 or len(rd) < 4:
            self.calib_failed = True
            self.mode = "monitoring" if self.thr is not None else "idle"
            return
        read = np.maximum(np.median(np.array(rd), axis=0), 0.5)
        track = np.maximum(np.median(np.array(tr), axis=0), read * self.factor)
        self.thr = np.sqrt(read * track)  # geometric midpoint between reading and tracking
        self.calib_info = {"read": round(float(np.median(read)), 1),
                           "track": round(float(np.median(track)), 1),
                           "thr": round(float(np.median(self.thr)), 1)}
        self.mode = "monitoring"
        self.drifting, self.move_since, self.calm_since = False, None, None
        self.reset_chunk()

    def pop_chunk(self):
        c = self.c
        self.reset_chunk()
        return {"steps": c["steps"], "distracted_frac": round(c["dist"] / max(c["steps"], 1), 2)}
