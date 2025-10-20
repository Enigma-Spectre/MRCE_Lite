# modules.py
import json
from typing import List, Optional, Dict, Any, Set
import dspy
import yaml

from signatures import (
    VibeSig,
    ShouldRespondSig,
    ExpertOutSig,
    SummarizeSig,
    MetaCriticSig,
    RoundMode,
    VibeLabel,
)
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

# ---- Expert classes ----

class VibeRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(VibeSig, temperature=0.0)

    def forward(self, query: str, history: dspy.History, goal: str, mode: RoundMode, guidance: str):
        return self.predict(system=SYSTEM_PERSONA, query=query, history=history, goal=goal, mode=mode, guidance=guidance)


class ExpertBase(dspy.Module):
    def __init__(self, persona_name: str, persona_text: str):
        super().__init__()
        self.persona_name = persona_name
        self.persona_text = persona_text
        self.gate = dspy.Predict(ShouldRespondSig, temperature=0.0)
        self.step = dspy.ChainOfThought(ExpertOutSig)

    def should_respond(self, query: str, history: dspy.History, goal: str, mode: RoundMode,
                       vibe: VibeLabel, hint: str = "", notes: str = ""):
        return self.gate(system=SYSTEM_PERSONA, persona=self.persona_name, persona_text=self.persona_text,
                         history=history, query=query, goal=goal, mode=mode, vibe=vibe,
                         hint=hint, notes=notes)

    def forward(self, query: str, history: dspy.History, goal: str, mode: RoundMode,
                vibe: VibeLabel, hint: str = "", notes: str = ""):
        return self.step(system=SYSTEM_PERSONA, persona=self.persona_name, persona_text=self.persona_text,
                         history=history, query=query, goal=goal, mode=mode, vibe=vibe,
                         hint=hint, notes=notes)


# Persona implementations with micro-guidelines
class Analyst(ExpertBase):
    def __init__(self):
        super().__init__("Analyst", "Terse, rigorous reasoning; favor equations and definitions.")


class Synthesizer(ExpertBase):
    def __init__(self):
        super().__init__("Synthesizer", "Generate concrete, non-redundant options; avoid fluff.")


class Critic(ExpertBase):
    def __init__(self):
        super().__init__("Critic", "Adversarial but fair; enumerate failure modes and safer alternatives.")


class Theorist(ExpertBase):
    def __init__(self):
        super().__init__("Theorist", "Favor formal models and proofs; abstract but precise.")


class Empiricist(ExpertBase):
    def __init__(self):
        super().__init__("Empiricist", "Ground claims in data or experiments; note unknowns.")


class Statistician(ExpertBase):
    def __init__(self):
        super().__init__("Statistician", "Quantify uncertainty; apply statistical tests and confidence.")


class SystemsEngineer(ExpertBase):
    def __init__(self):
        super().__init__("SystemsEngineer", "Think in components and interfaces; optimize reliability.")


class CounterexampleHunter(ExpertBase):
    def __init__(self):
        super().__init__("CounterexampleHunter", "Seek edge cases and contradictions; break assumptions.")


EXPERTS: List[ExpertBase] = [
    Analyst(),
    Synthesizer(),
    Critic(),
    Theorist(),
    Empiricist(),
    Statistician(),
    SystemsEngineer(),
    CounterexampleHunter(),
]

EXPERT_NAMES = [e.persona_name.lower() for e in EXPERTS]


def _load_expert_config(path: str = "experts.yaml") -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)

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
    def __init__(self, config_path: str = "experts.yaml", top_k: Optional[int] = None,
                 gate_min_conf: Optional[float] = None, gate_lambda: Optional[float] = None,
                 judge_temperature: float = 0.3):
        super().__init__()
        self.router = VibeRouter()
        self.experts = EXPERTS
        cfg = _load_expert_config(config_path)
        self.top_k = top_k if top_k is not None else cfg.get("top_k", 3)
        self.gate_min_conf = gate_min_conf if gate_min_conf is not None else cfg.get("gate_min_conf", 0.0)
        self.gate_lambda = gate_lambda if gate_lambda is not None else cfg.get("gate_lambda", 0.5)
        self.weights = cfg.get("weights", {})
        self.judge = dspy.MultiChainComparison("question -> answer", M=self.top_k, temperature=judge_temperature)

    def _weight(self, persona: str, mode: RoundMode) -> float:
        w = self.weights.get(persona, {})
        return w.get("base", 1.0) * w.get(mode, 1.0)

    def forward(self, query: str, history: dspy.History, goal: str, mode: RoundMode,
                router_guidance: str, expert_hints: Dict[str, str], want_payload: bool = False):
        vibe = self.router(query=query, history=history, goal=goal, mode=mode, guidance=router_guidance).vibe

        gating: List[Dict[str, Any]] = []
        for ex in self.experts:
            hint = expert_hints.get(ex.persona_name.lower(), "")
            gate = ex.should_respond(query=query, history=history, goal=goal, mode=mode, vibe=vibe, hint=hint)
            if gate.respond == "yes" and gate.confidence >= self.gate_min_conf:
                tags = {t.strip().lower() for t in (gate.coverage_tags or "").split(",") if t.strip()}
                score = gate.confidence * self._weight(ex.persona_name, mode)
                gating.append(dict(expert=ex, score=score, tags=tags, hint=hint))

        if not gating:
            # fallback: run analyst if all abstain
            gating.append(dict(expert=self.experts[0], score=1.0, tags=set(), hint=""))

        selected: List[Dict[str, Any]] = []
        candidates = gating[:]
        while candidates and len(selected) < self.top_k:
            best_idx = 0
            best_val = -1e9
            for i, c in enumerate(candidates):
                sim = 0.0
                if selected:
                    sim = max(_jaccard(c["tags"], s["tags"]) for s in selected)
                val = c["score"] - self.gate_lambda * sim
                if val > best_val:
                    best_val = val
                    best_idx = i
            selected.append(candidates.pop(best_idx))

        answers: List[str] = []
        labels: List[str] = []
        for item in selected:
            out = item["expert"](query=query, history=history, goal=goal, mode=mode, vibe=vibe, hint=item["hint"])
            answers.append(out.answer)
            labels.append(item["expert"].persona_name)

        try:
            self.judge.M = max(len(answers), 1)
        except Exception:
            pass

        try:
            judge_out, payload = _call_judge(self.judge, question=query, candidates=answers, labels=labels)
            ans, rat = _unpack_judge(judge_out)
        except Exception:
            judge_sig = dspy.Signature.from_string("question, a1, a2 -> answer, rationale")
            simple = dspy.Predict(judge_sig, temperature=0.0)
            a1 = answers[0] if answers else ""
            a2 = answers[1] if len(answers) > 1 else ""
            jo = simple(question=query, a1=a1, a2=a2)
            ans, rat, payload = jo.answer, jo.rationale, None

        pred = dspy.Prediction(vibe=vibe, candidates=answers, rationale=rat, answer=ans)
        pred.candidate_labels = labels
        if want_payload:
            pred.payload = payload
        return pred
