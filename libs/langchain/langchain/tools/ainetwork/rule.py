import json
from typing import Optional, Type, Union

from pydantic import BaseModel, Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.ainetwork.base import AINBaseTool, OperationType


class RuleSchema(BaseModel):
    type: OperationType = Field(...)
    path: str = Field(..., description="Blockchain reference path")
    config: Optional[dict] = Field(..., description="Rule config to be set")


class AINRuleOps(AINBaseTool):
    name: str = "AINruleOps"
    description: str = """# Syntax

Rule config is stored as a ".write" property on a path in the database using type SET operation. If you specify an operation of type GET, you can check the rules at that specific path in the database. In this case, the config parameter is ignored.
```
{<path>:{<to>:{<target_node>:{.write: <eval string to determine the write permission>}}}}
```
Its value is an javascript eval string that will be evaluated true or false to determine users' permission on the path whenever a transaction with value write operations on the path is submitted.

# Path Variables and Built-in Variables
The path can have path variables like "/transfer/$from/$to/value" to allow flexibility of rule expressions. In the same context, built-in variables are also provided by the system:

|Variable / Function|Members|Semantic|Example|
|---------------------|---------|----------------------------------------------------|-------------------------------------------------------------|
|auth|addr|Sender (signer) address|auth.addr === '$uid'|
|auth|fid|Caller (function) ID|auth.fid === '_transfer'|
|getValue(<db path>)||To get the value at the db path|getValue('/accounts/' + $user_addr + '/balance') >= 0|
|getRule(<db path>)||To get the rule at the db path|getRule('/apps/test_app')|
|getOwner(<db path>)||To get the owner at the db path|getOwner('/apps/test_app')|
|getFunction(<db path>)||To get the function at the db path|getFunction('/apps/test_app')|
|evalRule(<db path>, <value>, <auth>, <timestamp>)||To eval the rule config at the rule path|evalRule('/apps/test_app/posts/1', 'hello world', auth, currentTime)|
|evalOwner(<db path>, <permission>, <auth>)||To eval the owner config at the owner path|evalOwner('/apps/test_app/posts/1', 'write_owner', auth)|
|newData||The new data to be set at the given path|getValue('/accounts/' + $user_addr + '/balance') >= newData|
|data||The existing data at the given path|data !== null|
|currentTime||Current timestamp|currentTime <= $time + 24 * 60 * 60|
|lastBlockNumber||Last block number|lastBlockNumber > 10000|
|util||A collection of various utilities|
# Examples
Rule config can be set as the following example:

```
{transfer:{$from:{$to:{$key:{value:{.write:"auth.addr===$from&&!getValue('transfer/'+$from+'/'+$to+'/'+$key)&&getValue(util.getBalancePath($from))>=newData"}}}}},apps:{afan:{.write:"auth.addr==='0x12345678901234567890123456789012345678'",follow:{$uid:{.write:"auth.addr===$uid"}}}}}
```
There is no 'read' permission in data access. It means all network participants can read your data. To secure data on specific node path, users need to encrypt the data with their own private key.

# Application of Rule Config
Permission of a value write operation (e.g. SET_VALUE) is check as follows:
- When there are no rule config on the requested path, closest ancestor's rule config is applied
- If there are more than one path matched, the most specific rule config is applied
  - e.g. Among a) /apps/$app_id/$service, b) /apps/afan/$service, c) /apps/afan/wonny, c) is applied
- When the value of the write operation in request is an object, the operation is granted when the permission check succeeds on every path of object. For example, SET_VALUE operation is requested on /foo/bar with value { abc: "abc_val", def: "def_val" }, it should pass the permission check on /foo/bar, /foo/bar/abc, and /foo/bar/def.
- Rule config always overrides its ancestors' rule config
"""
    args_schema: Type[BaseModel] = RuleSchema

    async def _arun(
        self,
        type: OperationType,
        path: str,
        config: Optional[dict],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from ain.types import ValueOnlyTransactionInput

        try:
            if type is OperationType.SET:
                if config is None:
                    raise ValueError("'config' is required for SET operation.")

                res = await self.interface.db.ref(path).setRule(
                    transactionInput=ValueOnlyTransactionInput(value=config)
                )
            elif type is OperationType.GET:
                res = await self.interface.db.ref(path).getRule()
            else:
                raise ValueError(f"Unsupported 'type': {type}.")
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return f"{type(e).__name__}: {str(e)}"
