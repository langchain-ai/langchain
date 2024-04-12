import json

import pytest

from langchain_community.utilities.anysdk import AnySdkWrapper


class FakeSdk():
    def get_thing(self, thing_id):
        return {"id": thing_id}
    
    def post_thing(self, thing_id):
        return self._dummy_return(thing_id)
    
    def put_thing(self, thing_id):
        return self._dummy_return(thing_id)
    
    def delete_thing(self, thing_id):
        return self._dummy_return(thing_id)
    
    def _dummy_return(self, thing_id):
        return {"status": 200, "response": {"id": thing_id}}

client = FakeSdk()

anysdk = AnySdkWrapper(client=client)

def test_get_thing():
    assert json.loads(anysdk.run('get_thing', json.dumps({
            "thing_id": 123
        }))) == json.loads({"id": 123})

def test_post_thing():
    assert json.loads(anysdk.run('post_thing', json.dumps({
            "thing_id": 123
        }))) == json.loads({"status": 200, "response": {"id": 123}})

def test_put_thing():
    assert json.loads(anysdk.run('put_thing', json.dumps({
            "thing_id": 123
        }))) == json.loads({"status": 200, "response": {"id": 123}})

def test_delete_thing():
    assert json.loads(anysdk.run('delete_thing', json.dumps({
            "thing_id": 123
        }))) == json.loads({"status": 200, "response": {"id": 123}})

def test_no_hidden_methods():
    with pytest.rasises(AttributeError):
        anysdk.run('_dummy_return', json.dumps({"thing_id": 123}))