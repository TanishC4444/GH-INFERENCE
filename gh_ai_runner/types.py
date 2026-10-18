from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlsplit
from .limits import Limits


class JobStatus(str, Enum):
    DISPATCHED = "dispatched"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (self.COMPLETED, self.FAILED, self.CANCELLED)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 0
    delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be at least 0")
        if not math.isfinite(self.delay_seconds) or self.delay_seconds < 0:
            raise ValueError("delay_seconds must be at least 0")


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an OpenAI-compatible BYOK endpoint.

    The fixed ``secret_name`` identifies the package's dedicated GitHub Actions
    secret. The key itself is never included in dispatch inputs, logs, or artifacts.
    """

    name: str
    model: str
    base_url: str
    secret_name: str = "GH_AI_PROVIDER_API_KEY"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("provider name cannot be empty")
        if not self.model.strip():
            raise ValueError("provider model cannot be empty")
        url = urlsplit(self.base_url)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment or any(char.isspace() for char in self.base_url)):
            raise ValueError("provider base_url must be an HTTPS URL without credentials, query, or fragment")
        if self.secret_name != "GH_AI_PROVIDER_API_KEY":
            raise ValueError("Use the dedicated GH_AI_PROVIDER_API_KEY secret slot")

    @classmethod
    def groq(cls, model: str) -> ProviderConfig:
        return cls("groq", model, "https://api.groq.com/openai/v1")

    @classmethod
    def openai(cls, model: str) -> ProviderConfig:
        return cls("openai", model, "https://api.openai.com/v1")


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str
    model: str = "tinyllama"
    system: str = "You are a helpful assistant."
    max_tokens: Optional[int] = 512
    temperature: float = 0.7
    cache: bool = True
    n_ctx: Optional[int] = None
    provider: Optional[ProviderConfig] = None
    retry_policy: Optional[RetryPolicy] = None
    limits: Optional[Limits] = None
    model_options: dict = field(default_factory=dict)
    generation_options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ResourceUsage:
    cpu_count: int = 0
    memory_total_bytes: int = 0
    memory_available_bytes: int = 0
    disk_total_bytes: int = 0
    disk_free_bytes: int = 0
    peak_memory_bytes: int = 0
    duration_seconds: float = 0.0
    runner_os: str = ""


@dataclass(frozen=True)
class IntegrityInfo:
    algorithm: str = "sha256"
    request_sha256: str = ""
    output_sha256: str = ""


@dataclass
class AgentTimings:
    total_seconds: float = 0.0
    download_seconds: float = 0.0
    load_seconds: float = 0.0
    refinement_seconds: float = 0.0
    generation_seconds: float = 0.0
    remote_seconds: float = 0.0


@dataclass(frozen=True)
class InferenceResult:
    job_id: str
    correlation_id: str
    run_id: int
    output: str
    model: str
    provider: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    resources: ResourceUsage = field(default_factory=ResourceUsage)
    integrity: IntegrityInfo = field(default_factory=IntegrityInfo)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    timings: Optional[AgentTimings] = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("raw", None)
        return value


class JobFailedError(RuntimeError):
    def __init__(self, message: str, *, run_id: Optional[int] = None, conclusion: Optional[str] = None):
        super().__init__(message)
        self.run_id = run_id
        self.conclusion = conclusion


class JobCancelledError(JobFailedError):
    pass


class IntegrityError(JobFailedError):
    pass


@dataclass(frozen=True)
class JobRecord:
    correlation_id: str
    job_id: str
    attempt: int
    batch_id: Optional[str]
    owner: str
    repo_name: str
    status: str
    run_id: Optional[int]
    model: str
    provider: str
    prompt_sha256: str
    request_sha256: str
    output_sha256: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    duration_seconds: Optional[float]
    created_at: str
    updated_at: str
    error: Optional[str]
