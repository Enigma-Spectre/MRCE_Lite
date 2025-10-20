# modules.py
from typing import List, Optional, Dict
import dspy

from signatures import (
    VibeSig,
    AnalystSig,
    SynthSig,
    CriticSig,
    SummarizeSig,
    MetaCriticSig,
    RoundMode,
    VibeLabel,
    JudgeSig,
)
from config import SYSTEM_PERSONA

# ---- Modules ----

class VibeRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(VibeSig, temperature=0.0)
    def forward(self, query: str, history: dspy.History, goal: str, mode: RoundMode, guidance: str):
        return self.predict(system=SYSTEM_PERSONA, query=query, history=history, goal=goal, mode=mode, guidance=guidance)

class Analyst(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.ChainOfThought(AnalystSig)
    def forward(self, query: str, history: dspy.History, goal: str, mode: RoundMode, guidance: str):
        return self.step(system=SYSTEM_PERSONA, query=query, history=history, goal=goal, mode=mode, guidance=guidance)

class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.ChainOfThought(SynthSig)
    def forward(self, query: str, history: dspy.History, goal: str, mode: RoundMode, guidance: str):
        return self.step(system=SYSTEM_PERSONA, query=query, history=history, goal=goal, mode=mode, guidance=guidance)

class Critic(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.ChainOfThought(CriticSig)
    def forward(self, query: str, history: dspy.History, goal: str, mode: RoundMode, guidance: str):
        return self.step(system=SYSTEM_PERSONA, query=query, history=history, goal=goal, mode=mode, guidance=guidance)

class Judge(dspy.Module):
    def __init__(self, temperature: float = 0.3):
        super().__init__()
        self.step = dspy.ChainOfThought(JudgeSig, temperature=temperature)

    def forward(
        self,
        question: str,
        candidates: List[str],
        labels: Optional[List[str]] = None,
        want_payload: bool = False,
    ):
        blocks = []
        for i, text in enumerate(candidates, 1):
            label = labels[i - 1] if labels and i - 1 < len(labels) else f"Candidate {i}"
            blocks.append(f"[{label}]\n{text}")
        joined = "\n\n".join(blocks)
        out = self.step(system=SYSTEM_PERSONA, question=question, candidates=joined)
        ranking_line = str(getattr(out, "rankings", "")).strip()
        ranking = [lab.strip() for lab in ranking_line.split(">") if lab.strip()]
        best_label = str(getattr(out, "best", "")).strip()
        if not best_label and ranking:
            best_label = ranking[0]
        pred = dspy.Prediction(
            best_label=best_label,
            rankings=ranking,
            rationale=str(getattr(out, "rationale", "")),
        )
        if want_payload:
            pred.payload = joined
        return pred

class Summarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.Predict(SummarizeSig, temperature=0.0)
    def forward(self, history: dspy.History, latest_answer: str):
        return self.step(system=SYSTEM_PERSONA, history=history, latest_answer=latest_answer)

class MetaCritic(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.Predict(MetaCriticSig, temperature=0.0)
    def forward(self, query: str, goal: str, mode: RoundMode, chosen_vibe: VibeLabel,
                answer: str, rationale: str, summary_so_far: str):
        return self.step(system=SYSTEM_PERSONA, query=query, goal=goal, mode=mode, chosen_vibe=chosen_vibe,
                         answer=answer, rationale=rationale, summary_so_far=summary_so_far)

class MRCE_Lite(dspy.Module):
    def __init__(self, M: int = 3, judge_temperature: float = 0.3):
        super().__init__()
        self.router = VibeRouter()
        self.analyst = Analyst()
        self.synth = Synthesizer()
        self.critic = Critic()
        self.judge = Judge(temperature=judge_temperature)

    def forward(self, query: str, history: dspy.History, goal: str, mode: RoundMode,
                router_guidance: str, expert_hints: Dict[str, str], want_payload=False):
        vibe = self.router(query=query, history=history, goal=goal, mode=mode, guidance=router_guidance).vibe

        def hint(name): return expert_hints.get(name, "")

        if mode == "verify" or vibe in ("analytic", "plan"):
            ordered = [
                self.analyst(query=query, history=history, goal=goal, mode=mode, guidance=hint("analyst")).answer,
                self.critic(query=query, history=history, goal=goal, mode=mode, guidance=hint("critic")).answer,
                self.synth(query=query, history=history, goal=goal, mode=mode, guidance=hint("synth")).answer,
            ]
            labels = ["Analyst", "Critic", "Synth"]
        elif mode == "attack" or vibe == "critical":
            ordered = [
                self.critic(query=query, history=history, goal=goal, mode=mode, guidance=hint("critic")).answer,
                self.analyst(query=query, history=history, goal=goal, mode=mode, guidance=hint("analyst")).answer,
                self.synth(query=query, history=history, goal=goal, mode=mode, guidance=hint("synth")).answer,
            ]
            labels = ["Critic", "Analyst", "Synth"]
        else:  # explore / creative bias
            ordered = [
                self.synth(query=query, history=history, goal=goal, mode=mode, guidance=hint("synth")).answer,
                self.analyst(query=query, history=history, goal=goal, mode=mode, guidance=hint("analyst")).answer,
                self.critic(query=query, history=history, goal=goal, mode=mode, guidance=hint("critic")).answer,
            ]
            labels = ["Synth", "Analyst", "Critic"]

        judge_pred = self.judge(
            question=query, candidates=ordered, labels=labels, want_payload=want_payload
        )
        ranking = judge_pred.rankings
        chosen_label = judge_pred.best_label or (ranking[0] if ranking else labels[0])
        chosen_idx = labels.index(chosen_label) if chosen_label in labels else 0
        pred = dspy.Prediction(
            vibe=vibe,
            candidates=ordered,
            labels=labels,
            ranking=ranking,
            rationale=judge_pred.rationale,
            answer=ordered[chosen_idx],
        )
        if want_payload and getattr(judge_pred, "payload", None) is not None:
            pred.payload = judge_pred.payload
        return pred
