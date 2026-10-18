"""Optional diagnostics; Agent.run() itself returns plain text."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import AgentTimings, TokenUsage


@dataclass
class RefinementInfo:
    attempted: bool = False
    applied: bool = False
    original_tokens: int = 0
    refined_tokens: int = 0
    reason: str = "disabled"
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def tokens_saved(self):
        return max(0, self.original_tokens - self.refined_tokens) if self.applied else 0


@dataclass
class AgentResult:
    output: str
    model: str
    backend: str = "github"
    usage: TokenUsage = field(default_factory=TokenUsage)
    timings: Optional[AgentTimings] = None
    refinement: RefinementInfo = field(default_factory=RefinementInfo)
    model_reused: bool = False
    weights_cached: bool = False
    dropped_history_turns: int = 0
    finish_reason: str = "stop"
