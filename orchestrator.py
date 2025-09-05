# orchestrator.py
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import dspy
from signatures import RoundMode
from modules import MRCE_Lite, Summarizer, MetaCritic

DEFAULT_ULTIMATE_GOAL = "Reach irreducible truth or contradiction."

@dataclass
class OrchestratorState:
    history: dspy.History = field(default_factory=lambda: dspy.History(messages=[]))
    summary: str = ""
    goal: str = DEFAULT_ULTIMATE_GOAL
    mode: RoundMode = "verify"
    router_guidance: str = ""
    expert_hints: Dict[str, str] = field(default_factory=lambda: {"analyst": "", "synth": "", "critic": ""})
    round_idx: int = 0

class Orchestrator(dspy.Module):
    def __init__(self, max_rounds: int = 4):
        super().__init__()
        self.pipeline = MRCE_Lite()
        self.summarizer = Summarizer()
        self.critic = MetaCritic()
        self.max_rounds = max_rounds

    def forward(self, query: str, state: OrchestratorState, goal: Optional[str] = None,
                mode: Optional[RoundMode] = None, max_rounds: Optional[int] = None,
                verbose=True, print_judge_payload=False):
        if goal:
            state.goal = goal
        if mode:
            state.mode = mode
        if max_rounds is not None:
            self.max_rounds = max_rounds

        final = None
        trace: List[dict] = []

        for _ in range(self.max_rounds):
            state.round_idx += 1
            pred = self.pipeline(
                query=query,
                history=state.history,
                goal=state.goal,
                mode=state.mode,
                router_guidance=state.router_guidance,
                expert_hints=state.expert_hints,
                want_payload=print_judge_payload,
            )

            # Summarize + update
            summary_out = self.summarizer(history=state.history, latest_answer=pred.answer).summary
            state.summary = summary_out

            # Meta‑critic decision + hints
            meta = self.critic(
                query=query, goal=state.goal, mode=state.mode, chosen_vibe=pred.vibe,
                answer=pred.answer, rationale=pred.rationale, summary_so_far=state.summary
            )
            if meta.router_hint:
                state.router_guidance = (state.router_guidance + "\n" + meta.router_hint).strip()
            if meta.analyst_hint:
                state.expert_hints["analyst"] = (state.expert_hints["analyst"] + "\n" + meta.analyst_hint).strip()
            if meta.synth_hint:
                state.expert_hints["synth"] = (state.expert_hints["synth"] + "\n" + meta.synth_hint).strip()
            if meta.critic_hint:
                state.expert_hints["critic"] = (state.expert_hints["critic"] + "\n" + meta.critic_hint).strip()

            # Append to history (simple user/assistant abstraction)
            state.history.messages.append({"role": "user", "content": query})
            state.history.messages.append({"role": "assistant", "content": pred.answer})

            round_info = {
                "round": state.round_idx,
                "mode": state.mode,
                "goal": state.goal,
                "vibe": pred.vibe,
                "candidates": pred.candidates,
                "judge_rationale": pred.rationale,
                "judge_payload": getattr(pred, "payload", None),
                "meta": {
                    "stop_label": meta.stop_label,
                    "route_score": meta.route_score,
                    "quality_score": meta.quality_score,
                    "alignment_score": meta.alignment_score,
                    "router_hint": meta.router_hint,
                    "analyst_hint": meta.analyst_hint,
                    "synth_hint": meta.synth_hint,
                    "critic_hint": meta.critic_hint,
                },
                "summary": state.summary,
            }
            trace.append(round_info)

            if verbose:
                print(f"\n=== Round {state.round_idx} ===")
                print(f"Mode: {state.mode} | Goal: {state.goal}")
                print(f"Vibe: {pred.vibe}")
                print("\n-- Expert candidates (ordered) --")
                for i, c in enumerate(pred.candidates, 1):
                    print(f"[{i}] {c}\n")
                print("-- Judge rationale --")
                print(pred.rationale)
                if print_judge_payload and getattr(pred, 'payload', None) is not None:
                    print("-- Judge payload (completions) --")
                    try:
                        import json as _json
                        print(_json.dumps(pred.payload, indent=2)[:4000])
                    except Exception:
                        print(str(pred.payload)[:4000])
                print("-- Meta-critic --")
                print(f"stop_label={meta.stop_label} | route={meta.route_score:.2f} | quality={meta.quality_score:.2f} | align={meta.alignment_score:.2f}")
                if meta.router_hint: print(f"router_hint: {meta.router_hint}")
                if meta.analyst_hint: print(f"analyst_hint: {meta.analyst_hint}")
                if meta.synth_hint: print(f"synth_hint: {meta.synth_hint}")
                if meta.critic_hint: print(f"critic_hint: {meta.critic_hint}")
                print("-- Running summary --")
                print(state.summary)

            if meta.stop_label in ("irreducible_truth", "contradiction"):
                final = pred
                break

        if final is None:
            final = pred

        out = dspy.Prediction(
            answer=final.answer,
            rationale=final.rationale,
            vibe=final.vibe,
            goal=state.goal,
            mode=state.mode,
            rounds=state.round_idx,
            summary=state.summary,
            trace=trace,
        )
        return out
