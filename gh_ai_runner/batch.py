from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .job import AIJob
from .types import InferenceResult, JobStatus


class BatchJob:
    def __init__(self, jobs: list[AIJob], batch_id: Optional[str] = None) -> None:
        if not jobs:
            raise ValueError("a batch must contain at least one job")
        self.jobs = list(jobs)
        self.batch_id = batch_id or jobs[0].batch_id

    def refresh(self) -> dict[str, JobStatus]:
        return {job.job_id: job.refresh() for job in self.jobs}

    def wait(self, timeout: Optional[float] = None) -> BatchJob:
        started = time.monotonic()

        def wait_one(job: AIJob) -> None:
            remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - started))
            job.wait(timeout=remaining)

        with ThreadPoolExecutor(max_workers=min(8, len(self.jobs))) as executor:
            list(executor.map(wait_one, self.jobs))
        return self

    def results(self, timeout: Optional[float] = None) -> list[InferenceResult]:
        started = time.monotonic()

        def result_one(job: AIJob) -> InferenceResult:
            remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - started))
            return job.result(timeout=remaining)

        with ThreadPoolExecutor(max_workers=min(8, len(self.jobs))) as executor:
            return list(executor.map(result_one, self.jobs))

    def cancel(self) -> dict[str, bool]:
        with ThreadPoolExecutor(max_workers=min(8, len(self.jobs))) as executor:
            outcomes = list(executor.map(lambda job: job.cancel(), self.jobs))
        return {job.job_id: outcome for job, outcome in zip(self.jobs, outcomes)}


class ComparisonJob(BatchJob):
    def __init__(self, jobs: list[AIJob], models: list[str], batch_id: Optional[str] = None) -> None:
        super().__init__(jobs, batch_id=batch_id)
        self.models = list(models)

    def results_by_model(self, timeout: Optional[float] = None) -> dict[str, InferenceResult]:
        return dict(zip(self.models, self.results(timeout=timeout)))
