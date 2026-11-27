from unittest.mock import Mock, patch
import pytest
from gh_inference import ResidentSession


def runtime():
    llm=Mock()
    llm.tokenize.side_effect=lambda data,**kw:data.split()
    llm.create_chat_completion.return_value={'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}],'usage':{'prompt_tokens':4,'completion_tokens':2}}
    ns={'get_local_model':Mock(return_value=llm),'_LOADED_MODELS':{('qwen3b',8192):llm},'_MODEL_STATS':{('qwen3b',8192):{'load_seconds':1,'download_seconds':2}}}
    return llm,ns


def test_weights_reused_context_isolated():
    llm,ns=runtime()
    with patch('gh_ai_runner.resident._runtime',return_value=ns) as load:
        with ResidentSession('qwen3b') as session:
            session.complete('private first task',system='first')
            _,diagnostic=session.complete('second task',system='second')
            assert diagnostic['model_reused'] and diagnostic['context_reset']
            assert diagnostic['timings']['load_seconds']==0
            assert load.call_count==1
            assert llm.reset.call_count==2
            assert llm.create_chat_completion.call_args.kwargs['messages']==[{'role':'system','content':'second'},{'role':'user','content':'second task'}]
        llm.close.assert_called_once()


def test_context_rejected_before_generation_without_trimming():
    llm,ns=runtime()
    with patch('gh_ai_runner.resident._runtime',return_value=ns):
        session=ResidentSession('qwen3b')
        with pytest.raises(ValueError,match='No input was truncated'):
            session.complete('word '*7900,max_tokens=1000)
        llm.create_chat_completion.assert_not_called()


def test_json_schema_forwarded():
    llm,ns=runtime();schema={'type':'object','properties':{'id':{'enum':['e001','e002']}}}
    with patch('gh_ai_runner.resident._runtime',return_value=ns):
        ResidentSession('qwen3b').complete('select evidence',response_schema=schema)
    assert llm.create_chat_completion.call_args.kwargs['response_format']['schema']==schema


def test_import_no_runtime_loading():
    with patch('gh_ai_runner.resident._runtime') as load:
        ResidentSession('qwen3b')
        load.assert_not_called()
