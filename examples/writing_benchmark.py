"""Controlled writing tests. Models execute on GitHub; local checks inspect text only.

Run: GH_AI_TOKEN=... python examples/writing_benchmark.py
Results preserve raw answers; no repair, summarization, or model judging is applied.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from gh_inference import Agent, GitHubAIRunner, Limits, MODELS

MODELS_TO_TEST = {
    'magnum-v4-12b': 'bartowski/magnum-v4-12b-GGUF',
    'gemma-4-12b': 'unsloth/gemma-4-12b-it-GGUF',
    'gemma-3-12b': 'bartowski/google_gemma-3-12b-it-GGUF',
    'glitter-12b': 'duyntnet/Gemma-3-Glitter-12B-imatrix-GGUF',
    'qwen3.5-9b': 'unsloth/Qwen3.5-9B-GGUF',
    'gemma-4-styletune-12b': 'mradermacher/Gemma-4-12B-StyleTune-GGUF',
    'rocinante-x-12b': 'bartowski/TheDrummer_Rocinante-X-12B-v1-GGUF',
}

STYLE_MODELS = ['gemma-4-styletune-12b', 'rocinante-x-12b', 'glitter-12b', 'gemma-4-12b']
STYLE_CASES = {'style_scene': '''Write a short screenplay scene, around 180 to 240 words.
Mira, a cautious station engineer, and Sol, a pilot who uses dry humor to avoid emotional honesty, are trapped in a failing airlock. There is enough oxygen for only one person to reach the escape capsule. They used to be married. Neither wants to be the first to admit they still care.
Let their practical argument carry the emotional subtext. Give them distinct voices, concrete physical details, and a small change in their relationship by the end. Avoid stock romantic declarations and exposition about their history. Brief action lines are welcome. Return only the scene.'''}

CASES = {
    'dialogue': '''Write an English science-fiction scene under ALL these rules:
Exactly 12 nonempty lines, no title, commentary, narration, stage directions, or blank lines.
Every line is Speaker: spoken words. Alternate Mira then Sol, starting with Mira.
Only Mira and Sol speak. Every spoken line contains 3 to 14 whitespace-separated words.
Mira is a precise, cautious scientist. Sol is an impatient pilot who deflects fear with dry humor.
They are trapped behind a locked airlock door with five minutes of oxygen remaining.
Explicitly mention oxygen, the door, and five minutes somewhere in the dialogue.
Mira privately knows launch code 7319; never reveal that code in the output.
Do not use these words: destiny, whisper, whispered, suddenly.
The final line must be exactly: Sol: Then we leave together.
Make the conflict develop through dialogue and give each character a distinct voice.
Return only the 12 dialogue lines.''',
    'rhetoric': '''Write an English persuasive speech to a space station council asking them to repair oxygen filters instead of installing decorative lights.
Obey ALL rules: 140 to 180 whitespace-separated words total; exactly three paragraphs separated by blank lines; no title, bullets, speaker labels, or commentary.
The first sentence must be exactly: We cannot negotiate with vacuum.
Include this sentence exactly once: A promise is not a plan.
Include this concession exactly once: I understand the appeal of brighter halls.
Use no question marks. Do not use these words: destiny, delve, tapestry, testament.
Include a concrete requested action concerning oxygen filters, acknowledge the opposing preference, and use parallel phrasing without repeating whole paragraphs.
Return only the speech.''',
    'json_scene': '''Return ONLY one valid JSON object, without Markdown fences or commentary.
Its exact keys are scene_id, setting, characters, dialogue.
scene_id must be "airlock-7"; setting must be "locked airlock"; characters must be ["Mira", "Sol"] in that order.
dialogue must contain exactly six objects, each with ONLY the keys speaker and text, both strings.
Alternate speakers Mira then Sol, starting with Mira. Each text has 3 to 12 whitespace-separated words.
The last text must be exactly "Then we leave together."
The scene concerns a locked airlock and dwindling oxygen. Show Mira's measured caution and Sol's dry humor.
No stage directions. Never reveal launch code 7319. Do not use destiny, whisper, whispered, or suddenly.
All content must be in English.''',
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def forbidden_absent(text, words):
    return not re.search(r'\b(?:' + '|'.join(words) + r')\b', text, re.I)


def evaluate(case, text):
    """Mechanical constraints only. Voice, persuasion, and narrative quality need review."""
    checks = {'nonempty': bool(text.strip()), 'literal_private_code_absent': '7319' not in text}
    if case == 'rhetoric':
        paragraphs = re.split(r'\n\s*\n', text.strip())
        checks.update(
            word_count_140_180=140 <= len(text.split()) <= 180,
            three_paragraphs=len(paragraphs) == 3,
            required_opening=text.startswith('We cannot negotiate with vacuum.'),
            required_promise_once=text.count('A promise is not a plan.') == 1,
            required_concession_once=text.count('I understand the appeal of brighter halls.') == 1,
            no_questions='?' not in text,
            forbidden_words_absent=forbidden_absent(text, ['destiny', 'delve', 'tapestry', 'testament']),
            no_headings_or_bullets=not re.search(r'(?m)^\s*(?:#|\*|[-•]|\d+\.)', text),
        )
    elif case == 'dialogue':
        lines = text.strip().splitlines()
        matched = [re.fullmatch(r'(Mira|Sol):\s*(.+)', line) for line in lines]
        spoken = [m[2] for m in matched if m]
        checks.update(
            twelve_lines=len(lines) == 12,
            only_labeled_dialogue=bool(lines) and all(matched),
            alternating_speakers=len(matched) == 12 and all(m and m[1] == ('Mira' if i % 2 == 0 else 'Sol') for i, m in enumerate(matched)),
            line_word_limits=len(spoken) == 12 and all(3 <= len(line.split()) <= 14 for line in spoken),
            no_stage_markup=not re.search(r'[\[\]*]', text),
            final_line=bool(lines) and lines[-1] == 'Sol: Then we leave together.',
            forbidden_words_absent=forbidden_absent(text, ['destiny', 'whisper', 'whispered', 'suddenly']),
            mentions_oxygen=bool(re.search(r'\boxygen\b', text, re.I)),
            mentions_door=bool(re.search(r'\bdoor\b', text, re.I)),
            mentions_five_minutes=bool(re.search(r'\b(?:five|5)\s+minutes?\b', text, re.I)),
        )
    elif case == 'json_scene':
        try:
            obj = json.loads(text, object_pairs_hook=unique_object)
        except (ValueError, TypeError):
            obj = None
        valid = isinstance(obj, dict)
        if valid:
            checks['literal_private_code_absent'] = '7319' not in json.dumps(obj, ensure_ascii=False)
        dialogue = obj.get('dialogue') if valid else None
        valid_rows = (isinstance(dialogue, list) and len(dialogue) == 6 and
                      all(isinstance(row, dict) and set(row) == {'speaker', 'text'} and
                          isinstance(row['speaker'], str) and isinstance(row['text'], str) for row in dialogue))
        checks.update(
            valid_json=valid,
            exact_keys=valid and set(obj) == {'scene_id', 'setting', 'characters', 'dialogue'},
            scene_id=valid and obj.get('scene_id') == 'airlock-7',
            setting=valid and obj.get('setting') == 'locked airlock',
            characters=valid and obj.get('characters') == ['Mira', 'Sol'],
            dialogue_schema=bool(valid_rows),
            alternating_speakers=bool(valid_rows) and all(row['speaker'] == ('Mira' if i % 2 == 0 else 'Sol') for i, row in enumerate(dialogue)),
            line_word_limits=bool(valid_rows) and all(3 <= len(row['text'].split()) <= 12 for row in dialogue),
            final_line=bool(valid_rows) and dialogue[-1]['text'] == 'Then we leave together.',
            forbidden_words_absent=forbidden_absent(text, ['destiny', 'whisper', 'whispered', 'suddenly']),
            no_stage_markup=bool(valid_rows) and all(not re.search(r'[\[\]*]', row['text']) for row in dialogue),
        )
    else:
        raise ValueError('Unknown writing case')
    return {'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
            'all_mechanical_checks_pass': all(checks.values()),
            'violations': [name for name, passed in checks.items() if not passed],
            'word_count': len(text.split())}


def save(report, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODELS_TO_TEST))
    parser.add_argument('--suite', choices=['strict', 'style'], default='strict')
    parser.add_argument('--output', type=Path, default=Path('writing-results.json'))
    args = parser.parse_args()
    cases_to_run = STYLE_CASES if args.suite == 'style' else CASES
    args.models = args.models or (STYLE_MODELS if args.suite == 'style' else list(MODELS_TO_TEST)[:5])
    settings = {'n_ctx': 4096, 'max_tokens': 768, 'temperature': .7,
                'seed': 42, 'top_p': .9, 'threads': 4, 'runner_ram_gb': 16}
    report = {'validation_version': '1.1', 'started_at': datetime.now(timezone.utc).isoformat(), 'settings': settings,
              'prompts': cases_to_run, 'models': {},
              'method': 'One job and one model load per candidate; independent prompts, no memory, refinement, repairs, or grammar constraints. One sample per case. Mechanical checks are not a literary-quality score.'}
    limits = Limits(job_timeout_minutes=45, download_timeout_seconds=120)
    runner = GitHubAIRunner(metadata_path=None, runner_ram_gb=16, limits=limits,
                            completion_timeout=2800, run_discovery_timeout=300,
                            http_timeout=60, artifact_timeout=120, verbose=False)
    agents = {}
    for name in args.models:
        options = {'n_threads': 4, 'n_threads_batch': 4, 'n_batch': 256}
        if name in ('gemma-4-12b', 'gemma-4-styletune-12b', 'qwen3.5-9b'):
            options['chat_template_options'] = {'enable_thinking': False}
        try:
            agent = Agent(model=MODELS_TO_TEST[name], quantization='Q4_K_M', runner=runner,
                          instructions='Produce only the requested final writing artifact. Follow every explicit constraint. No reasoning, analysis, or commentary.',
                          remember=False, refine_prompt=False, track_time=True, n_ctx=4096,
                          max_tokens=768, temperature=.7, model_options=options,
                          generation_options={'seed': 42, 'top_p': .9})
            agents[name] = agent
            report['models'][name] = {'status': 'prepared', 'repository': MODELS_TO_TEST[name],
                                      'resolved_model': agent.model, 'model_spec': MODELS[agent.model],
                                      'model_options': options}
        except Exception as error:
            report['models'][name] = {'status': 'metadata_error', 'error': f'{type(error).__name__}: {error}'}
    save(report, args.output)
    runner.prepare()

    def run_one(name):
        agent = agents[name]
        begin = time.monotonic()
        outputs = agent.run_many(list(cases_to_run.values()), timeout=2700)
        cases = {case: {'output': output, 'evaluation': (evaluate(case, output) if args.suite == 'strict'
                         else {'word_count': len(output.split()), 'note': 'Literary quality requires review; no automatic style score.'}),
                         'diagnostics': asdict(result)}
                 for case, output, result in zip(cases_to_run, outputs, agent.last_results)}
        return {'status': 'completed', 'cases': cases, 'client_seconds': time.monotonic() - begin,
                'resources': asdict(agent.last_result.resources), 'actions_url': agent.last_job.actions_url}

    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as pool:
        pending = {pool.submit(run_one, name): name for name in agents}
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                name = pending.pop(future)
                try:
                    report['models'][name].update(future.result())
                    print(name, 'completed', flush=True)
                except Exception as error:
                    report['models'][name].update(status='execution_error', error=f'{type(error).__name__}: {error}')
                    print(name, 'failed', type(error).__name__, flush=True)
            for name, agent in agents.items():
                if agent.last_job:
                    report['models'][name].update(job_id=agent.last_job.job_id,
                                                  run_id=agent.last_job.run_id,
                                                  actions_url=agent.last_job.actions_url)
            report['updated_at'] = datetime.now(timezone.utc).isoformat()
            save(report, args.output)
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    save(report, args.output)


if __name__ == '__main__':
    main()
