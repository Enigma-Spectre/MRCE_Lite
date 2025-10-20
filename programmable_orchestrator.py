# programmable_orchestrator.py
"""DSPy agent wrapper around the orchestrator allowing programmatic use.

This agent exposes the existing :class:`Orchestrator` as a small DSPy module
that can be composed inside other DSPy programs.  It keeps the user-facing
API identical to ``Orchestrator.forward`` while adding a convenience method to
initialise state.

Example:
    agent = ProgrammableOrchestratorAgent()
    state = agent.init_state()
    result = agent(query="...", state=state)
"""

from __future__ import annotations

from typing import Optional

import dspy

from orchestrator import Orchestrator, OrchestratorState


class ProgrammableOrchestratorAgent(dspy.Module):
    """Thin wrapper that makes :class:`Orchestrator` programmable.

    Parameters
    ----------
    orchestrator: Optional[Orchestrator]
        Existing orchestrator instance.  If ``None`` a fresh one is created.
    """

    def __init__(self, orchestrator: Optional[Orchestrator] = None):
        super().__init__()
        self.orchestrator = orchestrator or Orchestrator()

    # Public API ---------------------------------------------------------
    def init_state(self) -> OrchestratorState:
        """Return a new :class:`OrchestratorState` object.

        Having a helper avoids forcing callers to know about the underlying
        dataclass and keeps the agent interface compact.
        """

        return OrchestratorState()

    def forward(self, query: str, state: OrchestratorState, **kwargs):
        """Run the underlying orchestrator.

        Additional keyword arguments are forwarded to ``Orchestrator.forward``
        making the agent easily customisable from a DSPy program.
        """

        return self.orchestrator.forward(query=query, state=state, **kwargs)

