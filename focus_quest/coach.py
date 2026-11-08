"""'Pip': short, plain check-in messages from the OpenAI API (offline fallback included)."""
import os

SYSTEM = ("You write short check-in messages for a study app that tracks focus from EEG. "
          "You get a numeric summary of the user's last few minutes. Write ONE calm, plain "
          "sentence of at most 10 words suggesting they refocus. No emoji, no exclamation "
          "marks, no cheerleading, no quotes.")
FALLBACK = ["Focus has dipped for a bit. Ready to refocus?",
            "Looks like you drifted. Back to the task?",
            "Quick reset: take a breath and refocus.",
            "Attention has been low for a while. Short break?"]


class Coach:
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.6")
        self.client, self.i = None, 0
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except ImportError:
                print("[coach] 'openai' package missing: pip install openai")

    def nudge(self, summ):
        """Return (text, source) where source is 'ai' or 'fallback'."""
        if self.client:
            prompt = (f"Last check-in window: distracted {summ['distracted_frac']:.0%} of the "
                      f"time, average engagement score {summ['mean_z']:+.1f} (0 = their own "
                      f"baseline, negative = drifting), signal noise {summ['artifact_frac']:.0%}.")
            try:
                r = self.client.responses.create(model=self.model, instructions=SYSTEM,
                                                 input=prompt)
                words = r.output_text.strip().strip('"').split()
                if words:
                    return " ".join(words[:12]), "ai"
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"[coach] OpenAI error, using fallback phrase: {e}")
        self.i += 1
        return FALLBACK[self.i % len(FALLBACK)], "fallback"
