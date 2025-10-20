# cli.py
import os
import sys
import json
import argparse
import threading

from config import configure_lm
from orchestrator import Orchestrator, DEFAULT_ULTIMATE_GOAL
from orchestrator import OrchestratorState

_HINT_KEY_ALIASES = {
    "synth": "synthesizer",
}


def _normalize_hint_key(key: str) -> str:
    if not isinstance(key, str):
        return key
    lowered = key.lower()
    return _HINT_KEY_ALIASES.get(lowered, lowered)
from programmable_orchestrator import ProgrammableOrchestratorAgent

_HINT_KEY_ALIASES = {
    "synth": "synthesizer",
}


def _normalize_hint_key(key: str) -> str:
    if not isinstance(key, str):
        return key
    lowered = key.lower()
    return _HINT_KEY_ALIASES.get(lowered, lowered)

def start_windows_esc_listener():
    if os.name != "nt":
        return None
    stop = threading.Event()
    def _watch():
        try:
            import msvcrt
        except Exception:
            return
        while not stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == "\x1b":  # ESC
                    print("\n[ESC detected] Exiting.")
                    os._exit(0)
    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return stop

def main():
    parser = argparse.ArgumentParser(description="MRCE‑lite DSPy (modular, verbose, hardened).")
    parser.add_argument("question", nargs="*", help="Your question (if omitted, enters interactive mode).")
    parser.add_argument("--goal", type=str, default=None, help="Ultimate goal for this query.")
    parser.add_argument("--mode", type=str, default=None, choices=["explore", "verify", "attack", "plan"], help="Round mode bias.")
    parser.add_argument("--max_rounds", type=int, default=3, help="Max rounds per query.")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-round printing.")
    parser.add_argument("--print_judge_payload", action="store_true", help="Print dict payload sent to judge.")
    parser.add_argument("--hints_cache", type=str, default=None, help="Persist router/expert hints as JSON.")
    parser.add_argument("--top_k", type=int, default=None, help="Number of experts to run (3-5).")
    parser.add_argument("--gate_min_conf", type=float, default=None, help="Min confidence for expert gating.")
    parser.add_argument("--gate_lambda", type=float, default=None, help="Diversity penalty for selection.")
    args = parser.parse_args()

    configure_lm()

    orchestrator = Orchestrator(
        max_rounds=args.max_rounds,
        top_k=args.top_k,
        gate_min_conf=args.gate_min_conf,
        gate_lambda=args.gate_lambda,
    )
    agent = ProgrammableOrchestratorAgent(orchestrator=orchestrator)
    state = agent.init_state()

    # Optional: load hints
    if args.hints_cache and os.path.exists(args.hints_cache):
        try:
            _h = json.load(open(args.hints_cache, "r"))
            state.router_guidance = _h.get("router_guidance", "")
            if isinstance(_h.get("expert_hints"), dict):
                for hint_key, hint_text in _h["expert_hints"].items():
                    norm_key = _normalize_hint_key(hint_key)
                    state.expert_hints[norm_key] = hint_text
            print(f"[loaded hints] {args.hints_cache}")
        except Exception as e:
            print(f"[warn] couldn't load hints: {e}")

    if args.question:
        q = " ".join(args.question).strip()
        try:
            pred = agent(
                query=q, state=state, goal=args.goal or DEFAULT_ULTIMATE_GOAL,
                mode=(args.mode or "verify"), max_rounds=args.max_rounds,
                verbose=not args.quiet, print_judge_payload=args.print_judge_payload
            )
        except Exception as e:
            msg = str(e)
            if "insufficient_quota" in msg or "RateLimitError" in msg or "429" in msg:
                print("\n[ERROR] Provider quota/rate-limit.")
                print("Quick switches (PowerShell examples):")
                print('  $env:DSPY_LM="anthropic/claude-3-haiku-20240307"; $env:ANTHROPIC_API_KEY="sk-ant-..."')
                print('  $env:DSPY_LM="gemini/gemini-2.5-pro-preview-03-25"; $env:GEMINI_API_KEY="..."')
                print('  $env:DSPY_LM="ollama_chat/llama3.2"; $env:DSPY_API_BASE="http://localhost:11434"; $env:DSPY_API_KEY=""')
                sys.exit(2)
            raise

        # Optional: save hints
        if args.hints_cache:
            try:
                json.dump(
                    {"router_guidance": state.router_guidance, "expert_hints": state.expert_hints},
                    open(args.hints_cache, "w"), indent=2
                )
                print(f"[saved hints] {args.hints_cache}")
            except Exception as e:
                print(f"[warn] couldn't save hints: {e}")

        print(f"\nRounds: {pred.rounds} | Mode: {pred.mode} | Vibe: {pred.vibe}")
        print("Goal:", pred.goal)
        print("\n--- Final Rationale ---\n", pred.rationale)
        print("\n--- Final Answer ---\n", pred.answer)
        print("\n--- Session Summary ---\n", pred.summary)
        return

    # Interactive loop
    print("MRCE‑lite (DSPy) — orchestrated multi‑round runs.")
    print("Quit options:")
    print("  • Windows: press ESC at any time.")
    print("  • Any OS : press Enter on an empty line, type 'esc'/'q' + Enter, or Ctrl+C.")
    stop_event = start_windows_esc_listener()

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("esc", "q", "quit", "exit"):
            break

        try:
            pred = agent(
                query=q, state=state, goal=state.goal, mode=state.mode,
                verbose=not args.quiet, print_judge_payload=args.print_judge_payload
            )
        except Exception as e:
            msg = str(e)
            if "insufficient_quota" in msg or "RateLimitError" in msg or "429" in msg:
                print("\n[ERROR] Provider quota/rate-limit. See env switch examples above.")
                continue
            raise

        if args.hints_cache:
            try:
                json.dump(
                    {"router_guidance": state.router_guidance, "expert_hints": state.expert_hints},
                    open(args.hints_cache, "w"), indent=2
                )
                print(f"[saved hints] {args.hints_cache}")
            except Exception as e:
                print(f"[warn] couldn't save hints: {e}")

        print(f"\nRounds: {pred.rounds} | Mode: {pred.mode} | Vibe: {pred.vibe}")
        print("Goal:", pred.goal)
        print("\n--- Final Rationale ---\n", pred.rationale)
        print("\n--- Final Answer ---\n", pred.answer)
        print("\n--- Session Summary ---\n", pred.summary)

    if stop_event is not None:
        stop_event.set()

if __name__ == "__main__":
    main()
