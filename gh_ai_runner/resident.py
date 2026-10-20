"""Resident local weights with stateless, independently scoped completions.

Install llama-cpp-python separately on the inference host. Importing this module
never loads a model, downloads weights, or dispatches a GitHub workflow.
"""
import hashlib
import json
import threading
import time

from .limits import Limits
from .models import MODELS
from .validation import _validate, validate_text


def _runtime(config):
    from .runner import INFERENCE_SCRIPT
    prefix = INFERENCE_SCRIPT.split('\nstarted = time.monotonic()\n', 1)[0]
    namespace = {'config': config, 'cpu_started': time.process_time()}
    exec(compile(prefix, '<resident-worker>', 'exec'), namespace)
    return namespace


class ResidentSession:
    """Reuse one model while resetting its context for every completion.

    A session is serialized by a lock. Separate processes/jobs do not share RAM.
    Never silently trims input or reduces the requested output allowance.
    """
    def __init__(self, model, *, n_ctx=8192, limits=None, model_options=None,
                 runner_ram_gb=16, runner_disk_gb=14):
        self.model, self.n_ctx = model, n_ctx
        self.limits = limits or Limits()
        self.options = dict(model_options or {})
        if set(self.options) & {"model_path", "n_ctx", "use_mmap", "use_mlock"}:
            raise ValueError("Model path, context and memory policy cannot be overridden in model_options")
        self.ram, self.disk = runner_ram_gb, runner_disk_gb
        _validate(model, 1, .25, n_ctx, runner_ram_gb=runner_ram_gb,
                  runner_disk_gb=runner_disk_gb, cache=False, limits=self.limits)
        self._namespace = None
        self._lock = threading.RLock()
        self._calls = 0

    def complete(self, prompt, *, system='', max_tokens=512, temperature=.25,
                 response_schema=None):
        validate_text(prompt, system, False, self.limits)
        _validate(self.model, max_tokens, temperature, self.n_ctx,
                  runner_ram_gb=self.ram, runner_disk_gb=self.disk,
                  cache=False, limits=self.limits)
        with self._lock:
            started = time.monotonic()
            config = {'model': self.model, 'n_ctx': self.n_ctx,
                      'limits': self.limits.to_dict(), 'cache_weights': False,
                      'model_options': self.options}
            if self._namespace is None:
                self._namespace = _runtime(config)
            llm = self._namespace['get_local_model'](self.model, self.n_ctx)
            messages = [{'role': 'system', 'content': system},
                        {'role': 'user', 'content': prompt}]
            input_tokens = sum(len(llm.tokenize(m['content'].encode(), add_bos=False))
                               for m in messages)
            reserve = max(512, self.limits.context_reserve_tokens)
            required = input_tokens + len(messages) * self.limits.message_overhead_tokens + reserve + max_tokens
            if required > self.n_ctx:
                raise ValueError('Input plus output and template reserve needs %d tokens; n_ctx=%d. No input was truncated.' % (required, self.n_ctx))
            # Reset token state, retaining loaded tensors. No previous messages are sent.
            llm.reset()
            options = {}
            if response_schema is not None:
                options['response_format'] = {'type': 'json_object', 'schema': response_schema}
            generation_started = time.monotonic()
            response = llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                                   temperature=temperature, **options)
            output = response['choices'][0]['message']['content'] or ''
            stats = self._namespace['_MODEL_STATS'][(self.model, self.n_ctx)]
            diagnostic = {
                'usage': response.get('usage', {}),
                'finish_reason': response['choices'][0].get('finish_reason'),
                'model_reused': self._calls > 0,
                'context_reset': True,
                'input_content_tokens': input_tokens,
                'context_required_with_reserve': required,
                'timings': {'total_seconds': time.monotonic()-started,
                            'generation_seconds': time.monotonic()-generation_started,
                            'load_seconds': stats['load_seconds'] if not self._calls else 0,
                            'download_seconds': stats['download_seconds'] if not self._calls else 0},
                'output_sha256': hashlib.sha256(output.encode()).hexdigest(),
                'model_revision': MODELS[self.model].get('url','').split('/resolve/')[-1].split('/')[0],
            }
            self._calls += 1
            return output, diagnostic

    def close(self):
        with self._lock:
            if self._namespace is not None:
                for llm in self._namespace['_LOADED_MODELS'].values():
                    llm.close()
                self._namespace['_LOADED_MODELS'].clear()
                self._namespace = None
            self._calls = 0

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
