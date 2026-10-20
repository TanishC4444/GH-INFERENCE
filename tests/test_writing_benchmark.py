import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('writing_benchmark', Path(__file__).parents[1] / 'examples' / 'writing_benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_valid_dialogue_and_exact_checks():
    lines = []
    for i in range(12):
        speaker = 'Mira' if i % 2 == 0 else 'Sol'
        text = 'The door is locked with five minutes of oxygen remaining.'
        if i == 11:
            text = 'Then we leave together.'
        lines.append(f'{speaker}: {text}')
    assert benchmark.evaluate('dialogue', '\n'.join(lines))['all_mechanical_checks_pass']
    lines[0] = 'Mira: whispered 7319'
    result = benchmark.evaluate('dialogue', '\n'.join(lines))
    assert 'literal_private_code_absent' in result['violations']
    assert 'forbidden_words_absent' in result['violations']


@pytest.mark.parametrize('text', ['not JSON', '```json\n{}\n```', '{"scene_id":"a","scene_id":"b"}', '[]'])
def test_bad_structured_output_is_not_silently_repaired(text):
    assert not benchmark.evaluate('json_scene', text)['checks']['valid_json']


def test_strict_json_scene():
    import json
    obj = {'scene_id': 'airlock-7', 'setting': 'locked airlock', 'characters': ['Mira', 'Sol'],
           'dialogue': [{'speaker': 'Mira' if i % 2 == 0 else 'Sol',
                         'text': 'Keep the oxygen flowing.' if i < 5 else 'Then we leave together.'} for i in range(6)]}
    assert benchmark.evaluate('json_scene', json.dumps(obj))['all_mechanical_checks_pass']
    obj['dialogue'][0]['text'] = 42
    assert 'dialogue_schema' in benchmark.evaluate('json_scene', json.dumps(obj))['violations']


def test_topic_synonyms_do_not_fail_unrequested_exact_phrase_check():
    paragraphs = [
        'We cannot negotiate with vacuum. ' + 'Maintain oxygen filtration equipment safely. ' * 9,
        'I understand the appeal of brighter halls. ' + 'Repair our essential equipment today. ' * 8,
        'A promise is not a plan. ' + 'Allocate the necessary maintenance funds. ' * 8,
    ]
    assert benchmark.evaluate('rhetoric', '\n\n'.join(paragraphs))['all_mechanical_checks_pass']


def test_json_escape_does_not_hide_the_literal_private_code():
    import json
    obj = {'scene_id': 'airlock-7', 'setting': 'locked airlock', 'characters': ['Mira', 'Sol'],
           'dialogue': [{'speaker': 'Mira' if i % 2 == 0 else 'Sol',
                         'text': 'Use 7319 right now.' if i == 0 else 'Then we leave together.'} for i in range(6)]}
    escaped = json.dumps(obj).replace('7319', r'\u0037\u0033\u0031\u0039')
    assert '7319' not in escaped
    assert 'literal_private_code_absent' in benchmark.evaluate('json_scene', escaped)['violations']
