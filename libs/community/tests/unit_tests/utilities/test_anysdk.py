import pytest
import json

from langchain_community.agent_toolkits.anysdk.toolkit import AnySdkToolkit
from langchain_community.utilities.anysdk import AnySdkWrapper\


class FakeSdk():
    def __init__(self):
        value = '123'

    def get_thing(self, thing_id):
        return {"id": thing_id}
    
    def post_thing(self, thing_id):
        return {"status": 200, "response": {"id": thing_id}}
    
    def put_thing(self, thing_id):
        return {"status": 200, "response": {"id": thing_id}}
    
    def delete_thing(self, thing_id):
        return {"status": 200, "response": {"id": thing_id}}
    
    def _shoudnt_wrap_this(self):
        return True

client = FakeSdk()

anysdk = AnySdkWrapper(client=client)

def test_get_thing():
    assert json.loads(anysdk.run('get_thing', json.dumps({"thing_id": 123}))) == json.loads({"id": 123})

def test_post_thing():
    assert json.loads(anysdk.run('post_thing', json.dumps({"thing_id": 123}))) == json.loads({"status": 200, "response": {"id": 123}})

def test_put_thing():
    assert json.loads(anysdk.run('put_thing', json.dumps({"thing_id": 123}))) == json.loads({"status": 200, "response": {"id": 123}})

def test_delete_thing():
    assert json.loads(anysdk.run('delete_thing', json.dumps({"thing_id": 123}))) == json.loads({"status": 200, "response": {"id": 123}})

def test_no_hidden_methods():
    with pytest.rasises(AttributeError) as e:
        anysdk.run('_shoudnt_wrap_this')