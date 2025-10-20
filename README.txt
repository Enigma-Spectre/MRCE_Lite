MRCE-lite (DSPy) — Modular, Robust, Verbose
====================================================

Ensemble reasoning with a Router + gated Expert pool + Judge + Orchestrator.
Experts include Analyst, Synthesizer, Critic, and additional personas (Theorist, Empiricist, Statistician, SystemsEngineer, CounterexampleHunter).
Hardened against DSPy MultiChainComparison API drift. Verbose per-round logs. Personas tuned for openai/gpt-4o-mini.

---------------------------------------------------------------------
File layout
---------------------------------------------------------------------
dspy_mrce_pkg/
  config.py         # LM config (provider-agnostic) + global SYSTEM_PERSONA
  signatures.py     # DSPy signatures + type aliases (RoundMode, VibeLabel)
  modules.py        # Router, Expert base/registry, Summarizer, MetaCritic, MRCE_Lite, judge helpers
  orchestrator.py   # OrchestratorState + Orchestrator (printing, trace)
  cli.py            # Entrypoint, flags, interactive loop, hints persistence

---------------------------------------------------------------------
Requirements
---------------------------------------------------------------------
- Python 3.10+
- Install DSPy:
    pip install dspy
  (Some envs use: pip install dspy-ai — either is fine.)

If you run a local model (Ollama, etc.), install it separately.

---------------------------------------------------------------------
Quick start (PowerShell, Windows)
---------------------------------------------------------------------
# 1) Activate your venv
# 2) Install DSPy
pip install dspy

# 3) Set key and run (defaults to openai/gpt-4o-mini)
$env:OPENAI_API_KEY="sk-..."
python .\dspy_mrce_pkg\cli.py "Is MMT valid, or are the assumptions shakey?" --mode verify --max_rounds 4 --print_judge_payload --hints_cache .\mrce_hints.json

Interactive mode (multi-run; ESC quits on Windows):
  python .\dspy_mrce_pkg\cli.py

---------------------------------------------------------------------
Quick start (bash/zsh, macOS/Linux)
---------------------------------------------------------------------
python -m venv .venv && source .venv/bin/activate
pip install dspy
export OPENAI_API_KEY="sk-..."
python dspy_mrce_pkg/cli.py "Is MMT valid, or are the assumptions shakey?" --mode verify --max_rounds 4 --print_judge_payload --hints_cache ./mrce_hints.json

---------------------------------------------------------------------
Switching models/providers (env vars; no code edits)
---------------------------------------------------------------------
OpenAI (default):
  DSPY_LM="openai/gpt-4o-mini"
  OPENAI_API_KEY must be set

Anthropic (fast/cheap judge):
  DSPY_LM="anthropic/claude-3-haiku-20240307"
  ANTHROPIC_API_KEY must be set

Google (Gemini):
  DSPY_LM="gemini/gemini-2.5-pro-preview-03-25"
  GEMINI_API_KEY must be set

Local (Ollama):
  DSPY_LM="ollama_chat/llama3.2"
  DSPY_API_BASE="http://localhost:11434"
  DSPY_API_KEY=""   # empty is fine for local

---------------------------------------------------------------------
What it prints (per round)
---------------------------------------------------------------------
- Round header: mode, goal, vibe
- Expert candidates (ordered by mode/vibe)
- Judge rationale (and optional judge payload dict if --print_judge_payload)
- Meta-critic scores: route/quality/alignment + hints per agent
- Running summary

---------------------------------------------------------------------
Flags
---------------------------------------------------------------------
--goal "..."                # turn goal (default: irreducible truth or contradiction)
--mode [explore|verify|attack|plan]
--max_rounds N
--quiet                     # suppress per-round prints
--print_judge_payload       # show dict sent to MultiChainComparison
--hints_cache path.json     # persist router/expert hints across runs
--top_k N                   # number of experts selected per round
--gate_min_conf X           # minimum confidence for gating (0-1)
--gate_lambda X             # diversity penalty for expert selection

---------------------------------------------------------------------
How it works
---------------------------------------------------------------------
- Router → labels the query with a vibe ∈ {analytic, creative, critical, plan}.
- ExpertScheduler → all personas run a cheap ShouldRespond gate. Top-k diverse experts (Analyst, Synthesizer, Critic, Theorist, Empiricist, Statistician, SystemsEngineer, CounterexampleHunter) produce structured answers.
- Judge → MultiChainComparison over selected candidates. If a DSPy version changes behavior, a fallback judge still returns an answer.
- Meta-Critic → scores routing, quality, alignment; returns hints that coach router/experts.
- Orchestrator → loops rounds until "irreducible truth" / "contradiction" (heuristic) or --max_rounds.

---------------------------------------------------------------------
Persistence
---------------------------------------------------------------------
Hints-only persistence (no dataset): pass --hints_cache path.json to store:
  - router_guidance
  - expert_hints per persona (analyst, synth, critic, ...)

True DSPy compilation (optional; not wired by default):
  - Collect a tiny trainset, run an optimizer like BootstrapFewShot().compile(...),
    then save and load compiled state to retain learned prompts/program across runs.

---------------------------------------------------------------------
Troubleshooting
---------------------------------------------------------------------
- 429 / insufficient_quota → Add credits, slow calls, or switch providers (DSPY_LM=..., relevant API_KEY).
- MultiChainComparison errors (missing 'completions', KeyError: 'answer') → Already handled; we send robust dicts and have a fallback judge.
- No output / sudden exit on Windows → You likely hit ESC (immediate exit by design).
- Costs/verbosity → Verbose mode prints (and spends tokens). Use --quiet after debugging.

---------------------------------------------------------------------
Extending
---------------------------------------------------------------------
Add a new expert:
  1) Create a signature in signatures.py (copy Analyst/Critic style).
  2) Add a module in modules.py (copy Analyst).
  3) Update ordering logic in MRCE_Lite.forward() (based on mode/vibe).
  4) If you want the meta-critic to coach it, add a hint slot in OrchestratorState.expert_hints.

Tighten the stop condition:
  Replace the heuristic with explicit claim extraction + contradiction check (new module), and gate on that.

---------------------------------------------------------------------
Example commands
---------------------------------------------------------------------
One-shot, verbose + payload, with hints persistence:
  python .\dspy_mrce_pkg\cli.py "Is MMT valid, or are the assumptions shakey?" ^
    --goal "Reach irreducible truth or contradiction." ^
    --mode verify ^
    --max_rounds 4 ^
    --print_judge_payload ^
    --hints_cache .\mrce_hints.json

Interactive (multi-question, stateful, hints persisted):
  python .\dspy_mrce_pkg\cli.py --hints_cache .\mrce_hints.json
  # Ask questions repeatedly; press ESC to quit.

---------------------------------------------------------------------
Design notes
---------------------------------------------------------------------
- Judge resilience: DSPy has changed MultiChainComparison inputs/outputs across versions. We supply redundant keys and a fallback so runs don’t break.
- Personas for 4o-mini: system guidance + structured deliverables → crisp, non-hallucinatory answers; respects mode/goal.
- Meta-feedback loop: short hints update router/experts each round; with --hints_cache, those hints carry across sessions without formal compilation.
