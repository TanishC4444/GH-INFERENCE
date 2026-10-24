from unittest.mock import Mock

import pytest

from gh_inference.__main__ import main, token_limit


def test_discussion_cli_accepts_dynamic_output(monkeypatch, capsys):
    factory = Mock()
    factory.return_value.discuss.return_value = [['reply']]
    monkeypatch.setattr('gh_inference.__main__.Agent', factory)
    monkeypatch.setattr('sys.argv', ['gh-inference', 'discuss', 'task', '--agents', '20',
                                   '--rounds', '3', '--max-tokens', 'auto', '--topology', 'ring'])
    main()
    assert factory.call_args.kwargs['max_tokens'] is None
    assert factory.call_args.kwargs['limits'].max_prompts == 60
    assert factory.return_value.discuss.call_args.kwargs['topology'] == 'ring'
    assert 'reply' in capsys.readouterr().out


@pytest.mark.parametrize('value', ['0', '-1', 'unlimited', '1.5'])
def test_invalid_token_limits(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        token_limit(value)
