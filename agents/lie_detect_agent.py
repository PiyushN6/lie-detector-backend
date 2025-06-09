from models.vision_model import VisionModel
from models.audio_model import AudioModel
from models.text_model import TextModel

class LieDetectAgent:
    def __init__(self):
        self.vision_model = VisionModel()
        self.audio_model = AudioModel()
        self.text_model = TextModel()
        self.thoughts = []

    def analyze(self, image=None, audio_file=None, text=None):
        self.thoughts.clear()
        scores = {}

        if image:
            vision_score = self.vision_model.predict_deception(image)
            scores['vision'] = vision_score
            self.thoughts.append(f"Facial analysis: {vision_score:.2f}")

        if audio_file:
            audio_score = self.audio_model.predict_deception(audio_file)
            scores['audio'] = audio_score
            self.thoughts.append(f"Audio analysis: {audio_score:.2f}")

        if text:
            text_score = self.text_model.predict_deception(text)
            scores['text'] = text_score
            self.thoughts.append(f"Text analysis: {text_score:.2f}")

        if not scores:
            return {"decision": "No input provided", "confidence": 0.0, "explanation": ""}

        avg_score = sum(scores.values()) / len(scores)
        decision = "Deceptive" if avg_score >= 0.5 else "Truthful"

        self.thoughts.append(f"Combined score: {avg_score:.2f}")
        self.thoughts.append(f"Final decision: {decision}")

        return {
            "decision": decision,
            "confidence": avg_score if decision == "Deceptive" else 1 - avg_score,
            "explanation": " | ".join(self.thoughts),
            "scores": scores
        }
