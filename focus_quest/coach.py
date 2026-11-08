"""'Pip': short, plain check-in messages from the OpenAI API (offline fallback included)."""
import os

SYSTEM = ("You write short check-in messages for a study app. It detects when the user is "
          "looking around (eyes or head moving a lot) instead of at their work. Write ONE "
          "calm, plain sentence of at most 10 words suggesting they refocus. No emoji, no "
          "exclamation marks, no cheerleading, no quotes.")
FALLBACK = ["You've been looking around a lot. Back to the task?",
            "Eyes back on your work?",
            "Quick reset: settle in and refocus.",
            "Lots of looking around lately. Ready to refocus?"]


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
            prompt = (f"In the last check-in window the user was looking around "
                      f"{summ['distracted_frac']:.0%} of the time.")
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
