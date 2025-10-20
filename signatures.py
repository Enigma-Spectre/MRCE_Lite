# signatures.py
import dspy
from typing import Literal

RoundMode = Literal["explore", "verify", "attack", "plan"]
VibeLabel = Literal["analytic", "creative", "critical", "plan"]


class ShouldRespondSig(dspy.Signature):
    """SYSTEM:\n{system}\n\nROLE: Expert gate. Decide whether this persona should answer.\nRETURN yes/no, confidence in [0,1], and comma‑separated coverage_tags."""
    system: str = dspy.InputField()
    persona: str = dspy.InputField(desc="Expert persona name")
    persona_text: str = dspy.InputField(desc="Persona description and micro-guidelines")
    history: dspy.History = dspy.InputField()
    query: str = dspy.InputField()
    goal: str = dspy.InputField()
    mode: RoundMode = dspy.InputField()
    vibe: VibeLabel = dspy.InputField()
    hint: str = dspy.InputField(desc="Extra guidance for this expert")
    notes: str = dspy.InputField(desc="Short persona notes or memory")
    respond: Literal["yes", "no"] = dspy.OutputField()
    confidence: float = dspy.OutputField()
    coverage_tags: str = dspy.OutputField(desc="Comma-separated topical tags")


class ExpertOutSig(dspy.Signature):
    """SYSTEM:\n{system}\n\nROLE: {persona}.\nSTYLE: follow persona_text.\nDELIVERABLE: structured answer with sections CLAIMS, EVIDENCE, ASSUMPTIONS, TESTS, RISKS. Max ~200 tokens."""
    system: str = dspy.InputField()
    persona: str = dspy.InputField(desc="Expert persona name")
    persona_text: str = dspy.InputField(desc="Persona description and micro-guidelines")
    history: dspy.History = dspy.InputField()
    query: str = dspy.InputField()
    goal: str = dspy.InputField()
    mode: RoundMode = dspy.InputField()
    vibe: VibeLabel = dspy.InputField()
    hint: str = dspy.InputField(desc="Extra guidance for this expert")
    notes: str = dspy.InputField(desc="Short persona notes or memory")
    answer: str = dspy.OutputField(desc="Structured text with required sections")

class VibeSig(dspy.Signature):
    """SYSTEM:\n{system}\n\nROLE: Router.\nTASK: Assign EXACTLY ONE label from {analytic, creative, critical, plan}.\n
    CRITERIA:\n- analytic: factual/precise answer or synthesis\n- creative: ideas/variants/brainstorm\n
    - critical: red-team, risks, failure modes\n- plan: step-by-step action plan\n
    OUTPUT: return only the label.\n"""
    system: str = dspy.InputField()
    query: str = dspy.InputField(desc="The user's request")
    history: dspy.History = dspy.InputField(desc="Conversation so far")
    goal: str = dspy.InputField(desc="Current goal for this turn")
    mode: RoundMode = dspy.InputField(desc="Round mode")
    guidance: str = dspy.InputField(desc="Router hints (short rules or examples)")
    vibe: VibeLabel = dspy.OutputField(desc="Single label")

class AnalystSig(dspy.Signature):
    """SYSTEM:\n{system}\n\nROLE: Analyst.\nSTYLE: terse, rigorous, citation‑free unless asked; prefer equations and definitions.\n
    DELIVERABLE: a compact answer aligned to the goal and mode.\nFORMAT:\n- CLAIMS\n- EVIDENCE (known/unknown)\n- CAVEATS\n"""
    system: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    query: str = dspy.InputField()
    goal: str = dspy.InputField()
    mode: RoundMode = dspy.InputField()
    guidance: str = dspy.InputField(desc="Hints to improve expert output")
    answer: str = dspy.OutputField()

class SynthSig(dspy.Signature):
    """SYSTEM:\n{system}\n\nROLE: Synthesizer.\nSTYLE: concrete, high‑signal bullets; vary options; avoid fluff.\n
    DELIVERABLE: 3–7 distinct options/ideas; prune repetition.\nFORMAT:\n- OPTIONS\n- HOW TO TEST\n- RISKS\n"""
    system: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    query: str = dspy.InputField()
    goal: str = dspy.InputField()
    mode: RoundMode = dspy.InputField()
    guidance: str = dspy.InputField(desc="Hints to improve expert output")
    answer: str = dspy.OutputField()

class CriticSig(dspy.Signature):
    """SYSTEM:\n{system}\n\nROLE: Red‑Team Critic.\nSTYLE: adversarial but fair; enumerate failure modes and kill‑shots.\n
    DELIVERABLE: prioritized risks & contradictions + safer alternatives.\nFORMAT:\n- FAILURE MODES\n- COUNTEREVIDENCE\n- SAFER PLANS\n"""
    system: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    query: str = dspy.InputField()
    goal: str = dspy.InputField()
    mode: RoundMode = dspy.InputField()
    guidance: str = dspy.InputField(desc="Hints to improve expert output")
    answer: str = dspy.OutputField()

class JudgeSig(dspy.Signature):
    """SYSTEM:\n{system}\n\nROLE: Judge. Compare candidates and rank them from best to worst.\nFORMAT:\n- BEST: top candidate label\n- RANKINGS: label1 > label2 > ...\n- RATIONALE: brief justification\n"""
    system: str = dspy.InputField()
    question: str = dspy.InputField()
    candidates: str = dspy.InputField(desc="Numbered candidate answers")
    best: str = dspy.OutputField(desc="Label of the top candidate")
    rankings: str = dspy.OutputField(desc="Candidate labels best-to-worst, separated by '>'")
    rationale: str = dspy.OutputField(desc="Why this ranking was chosen")

class SummarizeSig(dspy.Signature):
    """SYSTEM:\n{system}\n\nSummarize the dialogue so far + latest answer in <=120 words.\nKeep key facts, decisions, contradictions, open questions."""
    system: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    latest_answer: str = dspy.InputField()
    summary: str = dspy.OutputField()

class MetaCriticSig(dspy.Signature):
    """SYSTEM:\n{system}\n\nEvaluate routing + answer quality + goal alignment. Decide whether to stop.\n
    Return:\n- stop_label: continue | irreducible_truth | contradiction\n- route_score, quality_score, alignment_score in [0,1]\n
    - router_hint, analyst_hint, synth_hint, critic_hint: terse, actionable."""
    system: str = dspy.InputField()
    query: str = dspy.InputField()
    goal: str = dspy.InputField()
    mode: RoundMode = dspy.InputField()
    chosen_vibe: VibeLabel = dspy.InputField()
    answer: str = dspy.InputField()
    rationale: str = dspy.InputField()
    summary_so_far: str = dspy.InputField()
    stop_label: Literal["continue", "irreducible_truth", "contradiction"] = dspy.OutputField()
    route_score: float = dspy.OutputField()
    quality_score: float = dspy.OutputField()
    alignment_score: float = dspy.OutputField()
    router_hint: str = dspy.OutputField()
    analyst_hint: str = dspy.OutputField()
    synth_hint: str = dspy.OutputField()
    critic_hint: str = dspy.OutputField()
