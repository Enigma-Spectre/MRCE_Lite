# modules.py
import json
from typing import List, Optional, Dict, Any
import dspy

from signatures import VibeSig, AnalystSig, SynthSig, CriticSig, SummarizeSig, MetaCriticSig, RoundMode, VibeLabel
from config import SYSTEM_PERSONA

# ---- Judge helpers (version-safe) ----

def _as_completion_dicts(candidates: List[str], labels: Optional[List[str]] = None) -> List[dict]:
    out: List[dict] = []
    for i, text in enumerate(candidates, 1):
        text = (text or "").strip()
        name = labels[i-1] if labels and i-1 < len(labels) else f"Candidate {i}"
        gist = text.split("\n", 1)[0][:200]
        item = {
            "answer": text,
            "completion": text,
            "output": text,
            "response": text,
            "rationale": f"Source: {name}. One-line gist: {gist}",
            "reasoning": f"From {name}.",
        }
        out.append(item)
    return out

def _call_judge(judge, question: str, candidates: List[str], labels: Optional[List[str]] = None):
    try:
        payload = _as_completion_dicts(candidates, labels=labels)
        return judge(question=question, completions=payload), payload
    except TypeError:
        kwargs = {"question": question}
        for i, c in enumerate(candidates, 1):
            kwargs[f"reasoning_attempt_{i}"] = c
        return judge(**kwargs), None

def _unpack_judge(judge_out: Any):
    def get(obj, key, default=""):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    for key in ("answer", "best", "completion", "output", "response"):
        ans = get(judge_out, key, None)
        if ans:
            break
    else:
        ans = ""

    for key in ("rationale", "why", "critique", "explanation", "reason"):
        rat = get(judge_out, key, None)
        if rat:
            break
    else:
        rat = ""

    ans = str(ans) if ans is not None else ""
    rat = str(rat) if rat is not None else ""
    return ans, rat

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
        self.judge = dspy.MultiChainComparison("question -> answer", M=M, temperature=judge_temperature)

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

        try:
            judge_out, payload = _call_judge(self.judge, question=query, candidates=ordered, labels=labels)
            ans, rat = _unpack_judge(judge_out)
        except Exception:
            # Absolute fallback: pick best via a simple Predict judge
            judge_sig = dspy.Signature.from_string("""
                question, a1, a2, a3 -> answer, rationale
            """)
            simple = dspy.Predict(judge_sig, temperature=0.0)
            jo = simple(question=query, a1=ordered[0], a2=ordered[1], a3=ordered[2])
            ans, rat, payload = jo.answer, jo.rationale, None

        pred = dspy.Prediction(vibe=vibe, candidates=ordered, rationale=rat, answer=ans)
        if want_payload:
            pred.payload = payload
        return pred
