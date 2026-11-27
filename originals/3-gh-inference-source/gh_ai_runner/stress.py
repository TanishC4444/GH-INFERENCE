"""Bounded all-to-all agent benchmark. Model execution stays on GitHub."""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from .agent import Agent


def summarize(records):
    """Aggregate counters once per job; throughput includes prompt processing."""
    summary = {}
    for model in sorted({row['model'] for row in records}):
        rows = [row for row in records if row['model'] == model]
        good = [row for row in rows if 'result' in row]
        tokens = sum(row['result']['usage']['completion_tokens'] -
                     row['result']['refinement']['usage']['completion_tokens'] for row in good)
        seconds = sum(row['result']['timings']['generation_seconds'] for row in good)
        speeds = [(row['result']['usage']['completion_tokens'] -
                   row['result']['refinement']['usage']['completion_tokens']) /
                  row['result']['timings']['generation_seconds'] for row in good
                  if row['result']['timings']['generation_seconds'] > 0]
        cpus = [row['result']['resources']['cpu_seconds'] for row in good]
        energy = [row['estimated_energy_wh'] for row in good]
        latencies = [r['client_seconds'] for r in good if r.get('client_seconds') is not None]
        summary[model] = {
            'succeeded': len(good), 'failed': len(rows) - len(good),
            'completion_tokens': tokens,
            'generation_seconds': seconds,
            'fastest_completion_tokens_per_second': max(speeds, default=None),
            'completion_tokens_per_generation_second': tokens / seconds if seconds else None,
            'median_client_seconds': median(latencies) if latencies else None,
            'client_latency_samples': len(latencies),
            'cpu_seconds': sum(cpus) if cpus and all(v is not None for v in cpus) else None,
            'estimated_energy_wh': sum(energy) if energy and all(v is not None for v in energy) else None,
            'peak_process_memory_bytes': max([r['result']['resources']['peak_memory_bytes'] for r in good], default=0),
        }
        cpu = summary[model]['cpu_seconds']
        summary[model]['completion_tokens_per_cpu_second'] = tokens / cpu if cpu else None
    return summary


def stress_test(models=('qwen0.5b', 'llama', 'qwen', 'tinyllama'), *, agents=20,
                rounds=3, max_tokens=64, peer_chars=240, timeout=1800, topology="all",
                estimated_worker_watts=None, output='stress-results.json',
                agent_factory=Agent):
    """Run synchronized rounds, broadcasting every previous reply to every peer.

    One job per agent per round. Calls overlap; GitHub determines actual worker
    concurrency. Power is an optional user assumption, never a sensor reading.
    Failures are recorded and excluded from peer replies; later rounds continue.
    """
    for name, value in (('agents', agents), ('rounds', rounds), ('peer_chars', peer_chars)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ValueError('max_tokens must be a positive integer or None')
    if topology not in ('all', 'ring'):
        raise ValueError('topology must be all or ring')
    if not isinstance(models, (list, tuple)) or not models or any(not isinstance(m, str) or not m.strip() for m in models):
        raise ValueError('models must be a nonempty list or tuple of names')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    if estimated_worker_watts is not None and (isinstance(estimated_worker_watts, bool) or
            not isinstance(estimated_worker_watts, (int, float)) or
            not math.isfinite(estimated_worker_watts) or estimated_worker_watts <= 0):
        raise ValueError('estimated_worker_watts must be finite and positive')
    workers = []
    for index in range(agents):
        options = {'model': models[index % len(models)], 'track_time': True,
                   'remember': False, 'temperature': 0, 'max_tokens': max_tokens}
        if agent_factory is Agent and workers:
            options['runner'] = workers[0].runner
        worker = agent_factory(**options)
        worker.connect()
        workers.append(worker)
    report = {
        'agents': agents, 'rounds': rounds, 'max_tokens': max_tokens, 'topology': topology,
        'peer_chars': peer_chars, 'estimated_worker_watts': estimated_worker_watts,
        'metrics_scope': 'successful inference processes only; failed/unknown jobs excluded',
        'energy_method': 'assumed watts * worker wall seconds / 3600' if estimated_worker_watts else 'unavailable',
        'concurrency_method': 'overlapping client calls; hosted worker concurrency not measured',
        'records': [],
    }
    started_at = datetime.now(timezone.utc)
    report["started_at"] = started_at.isoformat()
    peers = {}

    def run_one(index, round_index, prompt):
        worker = workers[index]
        begin = time.monotonic()
        worker.last_job = None
        row = {'agent': index, 'round': round_index, 'model': worker.model}
        try:
            worker.run(prompt, timeout=timeout)
            row['result'] = asdict(worker.last_result)
            duration = worker.last_result.resources.duration_seconds
            row['estimated_energy_wh'] = (estimated_worker_watts * duration / 3600
                                          if estimated_worker_watts is not None else None)
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        job = worker.last_job
        row['actions_url'] = getattr(job, 'actions_url', None)
        row['job_id'] = getattr(job, 'job_id', None)
        row['run_id'] = getattr(job, 'run_id', None)
        row['client_seconds'] = time.monotonic() - begin
        return row

    with ThreadPoolExecutor(max_workers=agents) as pool:
        for round_index in range(rounds):
            task = ('Design a reliable, energy-efficient inference library. '
                    'Give one concrete improvement, challenge a peer claim when present, '
                    'and propose a measurable test. Keep your reply concise.\n')
            shared = '\n'.join(peers.values())
            futures = [pool.submit(run_one, i, round_index,
                       task + (peers.get((i - 1) % agents, '') if topology == 'ring' else shared))
                       for i in range(agents)]
            rows = [future.result() for future in futures]
            report['records'].extend(rows)
            peers = {row['agent']: f"Agent {row['agent']}: {row['result']['output'][:peer_chars]}"
                     for row in rows if 'result' in row}
            finished_at = datetime.now(timezone.utc)
            report['finished_at'] = finished_at.isoformat()
            report['wall_seconds'] = (finished_at - started_at).total_seconds()
            report['summary'] = summarize(report['records'])
            if output is not None:
                path = Path(output)
                temporary = path.with_suffix(path.suffix + '.tmp')
                temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
                temporary.replace(path)
    return report
