"""
NeuroFocus MVP.

Reads the NeuroPawn LSL stream, detects calibrated blink and jaw-clench
artifacts, and provides a simplified dyslexia-friendly essay workspace.

Artifact control is intentionally separated from the experimental attention
estimate. EEG artifacts can control commands; they do not prove attention or
thought.
"""

import os
import shutil
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from collections import deque

import numpy as np
from pylsl import StreamInlet, resolve_streams


STREAM_NAME = "NeuroPawnKnight"
POLL_MS = 40
EXPERIMENTAL_ATTENTION_WINDOW = 10


class ArtifactDetector:
    def __init__(self):
        self.reset_calibration()
        self.last_blink = 0.0
        self.blink_times = deque(maxlen=4)
        self.last_action = 0.0

    def reset_calibration(self):
        self.baseline_samples = []
        self.blink_samples = []
        self.clench_samples = []
        self.baseline = 0.0
        self.blink_threshold = float("inf")
        self.clench_threshold = float("inf")

    def add_baseline(self, sample):
        self.baseline_samples.append(float(np.max(np.abs(sample[:8]))))

    def add_blink(self, sample):
        self.blink_samples.append(float(np.max(np.abs(sample[:8]))))

    def add_clench(self, sample):
        self.clench_samples.append(float(np.max(np.abs(sample[:8]))))

    def calibrate(self):
        if len(self.baseline_samples) < 50:
            return False, "Keep the baseline recording running for a little longer."

        baseline = np.asarray(self.baseline_samples)
        self.baseline = float(np.median(baseline))
        baseline_high = float(np.percentile(baseline, 95))

        if self.blink_samples:
            blink = np.asarray(self.blink_samples)
            blink_peak = float(np.percentile(blink, 90))
            self.blink_threshold = max(baseline_high * 1.35, blink_peak * 0.6)

        if self.clench_samples:
            clench = np.asarray(self.clench_samples)
            clench_peak = float(np.percentile(clench, 90))
            self.clench_threshold = max(baseline_high * 1.8, clench_peak * 0.55)

        return True, (
            f"Calibration complete. Blink threshold: {self.blink_threshold:.0f}, "
            f"clench threshold: {self.clench_threshold:.0f}."
        )

    def analyze(self, sample):
        now = time.monotonic()
        magnitude = float(np.max(np.abs(sample[:8])))

        if magnitude > self.clench_threshold and now - self.last_action > 1.5:
            self.last_action = now
            return "clench"

        if magnitude > self.blink_threshold and now - self.last_action > 0.35:
            self.last_blink = now
            self.blink_times.append(now)
            while self.blink_times and now - self.blink_times[0] > 1.0:
                self.blink_times.popleft()
            if len(self.blink_times) >= 2 and now - self.last_action > 0.35:
                self.last_action = now
                self.blink_times.clear()
                return "double_blink"

        return None


class NeuroFocusApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NeuroFocus")
        self.geometry("1180x760")

        self.detector = ArtifactDetector()
        self.inlet = None
        self.buffer = deque(maxlen=125 * EXPERIMENTAL_ATTENTION_WINDOW)
        self.calibration_stage = None
        self.connected = False
        self._build_ui()
        self.refresh_stream()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(POLL_MS, self.update_stream)

    def _build_ui(self):
        self.configure(bg="#f4f7fb")

        header = ttk.Frame(self)
        header.pack(fill="x", padx=18, pady=12)
        ttk.Label(header, text="NeuroFocus", font=("Segoe UI", 22, "bold")).pack(
            side="left"
        )
        ttk.Label(
            header,
            text="Essay workspace for focused writing",
            foreground="#52617a",
        ).pack(side="left", padx=12)

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        left = ttk.Frame(main, padding=10)
        right = ttk.Frame(main, padding=10)
        main.add(left, weight=2)
        main.add(right, weight=1)

        ttk.Label(left, text="Essay title", font=("Segoe UI", 11, "bold")).pack(
            anchor="w"
        )
        self.title_entry = ttk.Entry(left, font=("Segoe UI", 12))
        self.title_entry.pack(fill="x", pady=(4, 12))

        ttk.Label(left, text="Draft", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.editor = tk.Text(
            left,
            wrap="word",
            font=("Segoe UI", 13),
            padx=18,
            pady=18,
            relief="flat",
        )
        self.editor.pack(fill="both", expand=True, pady=(4, 10))

        task_frame = ttk.LabelFrame(left, text="Small next steps", padding=8)
        task_frame.pack(fill="x")
        self.tasks = [
            tk.BooleanVar(value=False),
            tk.BooleanVar(value=False),
            tk.BooleanVar(value=False),
        ]
        for text, var in zip(
            ("Write the main idea", "Add one supporting point", "Read and improve two sentences"),
            self.tasks,
        ):
            ttk.Checkbutton(task_frame, text=text, variable=var).pack(anchor="w")

        eeg_frame = ttk.LabelFrame(right, text="NeuroPawn controls", padding=10)
        eeg_frame.pack(fill="x")

        self.stream_var = tk.StringVar()
        self.stream_combo = ttk.Combobox(eeg_frame, textvariable=self.stream_var, state="readonly")
        self.stream_combo.pack(fill="x")
        button_row = ttk.Frame(eeg_frame)
        button_row.pack(fill="x", pady=8)
        ttk.Button(button_row, text="Refresh", command=self.refresh_stream).pack(
            side="left"
        )
        ttk.Button(button_row, text="Connect", command=self.toggle_connect).pack(
            side="left", padx=6
        )

        calibrate_row = ttk.Frame(eeg_frame)
        calibrate_row.pack(fill="x")
        ttk.Button(
            calibrate_row,
            text="Baseline 4s",
            command=lambda: self.start_calibration("baseline"),
        ).pack(side="left")
        ttk.Button(
            calibrate_row,
            text="Blink 4s",
            command=lambda: self.start_calibration("blink"),
        ).pack(side="left", padx=4)
        ttk.Button(
            calibrate_row,
            text="Clench 4s",
            command=lambda: self.start_calibration("clench"),
        ).pack(side="left")
        ttk.Button(
            eeg_frame, text="Finish calibration", command=self.finish_calibration
        ).pack(fill="x", pady=8)

        self.status_var = tk.StringVar(value="LSL disconnected")
        ttk.Label(eeg_frame, textvariable=self.status_var, wraplength=330).pack(
            fill="x"
        )
        ttk.Button(
            eeg_frame,
            text="Test double blink",
            command=lambda: self.trigger_action("double_blink"),
        ).pack(fill="x", pady=(8, 0))
        ttk.Button(
            eeg_frame,
            text="Test jaw clench",
            command=lambda: self.trigger_action("clench"),
        ).pack(fill="x", pady=4)

        ai_frame = ttk.LabelFrame(right, text="Writing assistant", padding=10)
        ai_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.ai_output = tk.Text(
            ai_frame,
            height=14,
            wrap="word",
            font=("Segoe UI", 11),
            relief="flat",
        )
        self.ai_output.pack(fill="both", expand=True)

        actions = ttk.Frame(ai_frame)
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(actions, text="Brainstorm", command=lambda: self.ask_ai("brainstorm")).pack(
            fill="x", pady=2
        )
        ttk.Button(actions, text="Organize", command=lambda: self.ask_ai("organize")).pack(
            fill="x", pady=2
        )
        ttk.Button(actions, text="Improve writing", command=lambda: self.ask_ai("improve")).pack(
            fill="x", pady=2
        )
        ttk.Button(actions, text="Save idea", command=self.save_idea).pack(
            fill="x", pady=2
        )

    def refresh_stream(self):
        streams = resolve_streams(wait_time=0.5)
        names = [
            f"{stream.name()} ({stream.channel_count()}ch)"
            for stream in streams
        ]
        self.stream_combo["values"] = names
        if names:
            preferred = next(
                (i for i, stream in enumerate(streams) if stream.name() == STREAM_NAME),
                0,
            )
            self.stream_combo.current(preferred)
            self.streams = streams
        else:
            self.streams = []
            self.status_var.set("No LSL stream found. Start the NeuroPawn streamer.")

    def toggle_connect(self):
        if self.inlet is not None:
            try:
                self.inlet.close_stream()
            except Exception:
                pass
            self.inlet = None
            self.connected = False
            self.status_var.set("LSL disconnected")
            return

        index = self.stream_combo.current()
        if index < 0 or index >= len(getattr(self, "streams", [])):
            messagebox.showerror("No stream", "No NeuroPawn LSL stream is selected.")
            return
        self.inlet = StreamInlet(self.streams[index], max_buflen=10)
        self.connected = True
        self.status_var.set("LSL connected. Baseline calibration is recommended.")

    def start_calibration(self, stage):
        self.calibration_stage = stage
        if stage == "baseline":
            self.detector.reset_calibration()
            message = "Stay relaxed and keep your face neutral for 4 seconds."
        elif stage == "blink":
            message = "Blink normally, then deliberately double-blink near the end."
        else:
            message = "Clench your jaw 3 to 4 times during these 4 seconds."
        self.status_var.set(message)
        self.after(4000, lambda: self.end_calibration(stage))

    def end_calibration(self, stage):
        self.calibration_stage = None
        self.status_var.set(f"{stage.capitalize()} samples recorded. Continue calibration.")

    def finish_calibration(self):
        ok, message = self.detector.calibrate()
        self.status_var.set(message)
        if not ok:
            messagebox.showwarning("Calibration incomplete", message)

    def update_stream(self):
        if self.inlet is not None:
            chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=64)
            if chunk:
                samples = np.asarray(chunk, dtype=np.float64)
                for sample in samples:
                    if self.calibration_stage == "baseline":
                        self.detector.add_baseline(sample)
                    elif self.calibration_stage == "blink":
                        self.detector.add_blink(sample)
                    elif self.calibration_stage == "clench":
                        self.detector.add_clench(sample)
                    self.buffer.append(sample[:8])
                    action = self.detector.analyze(sample)
                    if action:
                        self.trigger_action(action)
        self.after(POLL_MS, self.update_stream)

    def trigger_action(self, action):
        if action == "clench":
            self.ask_ai("brainstorm")
        elif action == "double_blink":
            self.dismiss_focus_reminder()

    def dismiss_focus_reminder(self):
        self.status_var.set("AI reminder dismissed with double blink.")

    def save_idea(self):
        text = self.editor.get("sel.first", "sel.last") if self.editor.tag_ranges("sel") else ""
        if not text.strip():
            text = self.editor.get("1.0", "end").strip()[-300:]
        if text:
            self.ai_output.insert("end", "\n\nSaved idea:\n" + text + "\n")
            self.ai_output.see("end")

    def ask_ai(self, mode):
        text = self.editor.get("1.0", "end").strip()
        title = self.title_entry.get().strip() or "Untitled essay"
        prompts = {
            "brainstorm": "Brainstorm three accessible ideas and a simple thesis for this essay.",
            "organize": "Turn this draft into a clear three-part outline.",
            "improve": "Improve the sentence structure while preserving the writer's meaning.",
        }
        prompt = (
            f"Help a dyslexic student write an essay titled '{title}'. "
            f"{prompts[mode]} Keep the response supportive, concise, and easy to scan.\n\n{text}"
        )
        self.ai_output.insert("end", "\nThinking...\n")
        self.ai_output.see("end")

        def run():
            result = self.run_claude(prompt)
            self.after(0, lambda: self.ai_output.insert("end", "\n" + result + "\n"))
            self.after(0, self.ai_output.see, "end")

        threading.Thread(target=run, daemon=True).start()

    @staticmethod
    def run_claude(prompt):
        executable = os.getenv("CLAUDE_CODE_PATH") or shutil.which("claude")
        if not executable:
            return (
                "Claude Code is not installed or is not on PATH. Add Claude Code to PATH, "
                "or set CLAUDE_CODE_PATH. Until then, use these local writing prompts:\n"
                "- What is the main point in one sentence?\n"
                "- Add one reason that supports the point.\n"
                "- Add one example from personal experience or research."
            )
        try:
            process = subprocess.run(
                [executable, "-p", prompt],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if process.stdout.strip():
                return process.stdout.strip()
            return process.stderr.strip() or "Claude Code returned no response."
        except subprocess.TimeoutExpired:
            return "Claude Code timed out. Please try a shorter request."
        except OSError as exc:
            return f"Could not run Claude Code: {exc}"

    def on_close(self):
        if self.inlet is not None:
            try:
                self.inlet.close_stream()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    NeuroFocusApp().mainloop()
