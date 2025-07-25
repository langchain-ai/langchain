import re
import os
import json
from typing import Literal, Optional, Tuple, Union, Annotated
from pydantic import BaseModel, Field, PositiveInt, ValidationInfo, field_validator, ConfigDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Ensure you have your OPENAI_API_KEY set as an environment variable
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set.")
    
# Dummy placeholder since this isn't a real LangGraph state injection
def InjectedState(d: dict):
    return {}

# --- Pydantic Models from the GitHub Issue ---

time_fmt = "%Y-%m-%d %H:%M:%S"
time_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"

# Forward-declare nested models for Pydantic
class DataSoilDashboardQueryPayloadQueryParam:
    pass

class DataSoilDashboardQueryPayloadTimeShift(BaseModel):
    shiftInterval: list[PositiveInt] = Field(description="Each element in the array represents a time offset relative to the query timestamp for individual time comparison analysis. If time comparison analysis dose not described, keep it **VOID**.",max_length=2,default=[])
    timeUnit: Literal["DAY"] = Field(default="DAY",description="The unit of specific comparison time offset. This is the description about each value of unit: Unit **DAY** represents one day.")

class DataSoilDashboardQueryPayloadQueryParamWhereFilter(BaseModel):
    field: str = Field(description="The dimension **CODE** in the selected dimension list that requires enums filtering or pattern filtering.")
    operator: Literal["IN", "NI", "LIKE", "NOT_LIKE"] = Field(description="Operators for enums filtering or pattern filtering.")
    value: list[str] = Field(description="If for enums filtering, every element represents th practical enums of the dimension. Otherwise for pattern filtering, only **one** element is required and it represents a wildcard pattern.",min_length=1)
    
    @field_validator("field")
    def field_block(cls, v: str, info: ValidationInfo) -> str:
        if v == "dt":
            raise ValueError("Instruction: The time filtering should be described in 'time' field, not in the 'filters' field.")
        return v
    
    @field_validator("value")
    def value_block(cls, v: Optional[list[str]], info: ValidationInfo) -> Optional[list[str]]:
        if info.data.get("operator") in {"LIKE", "NOT_LIKE"} and v and len(v) > 1:
            raise ValueError("Instruction: For pattern filtering, the size of 'value' in 'where' must be **ONE**.")
        return v

class DataSoilDashboardQueryPayloadQueryParamWhere(BaseModel):
    time: list[Union[str, int]] = Field(description=f"The target time range...", min_length=2, max_length=2)
    filters: list[DataSoilDashboardQueryPayloadQueryParamWhereFilter] = Field(description="Enums filtering or pattern filtering condition...")
    relation: Literal["AND"] = Field(description="Boolean relationships between filters...")

    @field_validator("time")
    def time_format_block(cls, v: list[Union[int, str]], info: ValidationInfo) -> list[Union[int, str]]:
        if isinstance(v[0], str) and not re.search(time_pattern, v[0]):
            raise ValueError(f"Instruction: the start time of time range must be formatted as **{time_fmt}**")
        if isinstance(v[1], str) and not re.search(time_pattern, v[1]):
            raise ValueError(f"Instruction: the end time of time range must be formatted as **{time_fmt}**")
        return v

class DataSoilDashboardQueryPayloadQueryParamOrderBy(BaseModel):
    field: str = Field(description="The metric **CODE** in the selected metric list that requires metric sorting.")
    direction: Literal["ASC", "DESC"] = Field(description="Sorting direction for specified metric.")
    shift: int = Field(default=0)
    limit: int = Field(description="The number of rows to return...", default=50)

class DataSoilDashboardQueryPayloadQueryParamGroupBy(BaseModel):
    field: str = Field(description="The dimension **CODE** in the selected dimension list for dimension grouping analysis.")
    extendFields: list[str] = Field(default=[])
    orderBy: Optional[DataSoilDashboardQueryPayloadQueryParamOrderBy] = Field(description="Sorting config for query results...", default=None)

class DataSoilDashboardQueryPayloadQueryParam(BaseModel):
    queryType: Literal["DETAIL_TABLE"] = Field(description="This is the description about queryType...")
    interval: Literal["BY_ONE_MINUTE", "BY_FIVE_MINUTE", "BY_HOUR", "BY_DAY", "BY_WEEK", "BY_MONTH", "SUM"] = Field(description="The time granularity for time-based grouping analysis.")
    resultField: list[str] = Field(default=[])
    where: DataSoilDashboardQueryPayloadQueryParamWhere = Field(description="Filtering condition for dimensions.")
    groupBy: list[DataSoilDashboardQueryPayloadQueryParamGroupBy] = Field(description="A list of dimensions grouping analysis info...")
    orderBy: DataSoilDashboardQueryPayloadQueryParamOrderBy = Field(description="Sorting config for query results...")
    heavyQuery: bool = Field(default=False)
    
    @field_validator("groupBy")
    def groupBy_block(cls, v: list[DataSoilDashboardQueryPayloadQueryParamGroupBy], info: ValidationInfo) -> list[DataSoilDashboardQueryPayloadQueryParamGroupBy]:
        if "dt" in {e.field for e in v}:
            if info.data.get("interval") == "SUM":
                raise ValueError("Instruction: the interval can not be **SUM** when **time-based grouping is required**.")
        else:
            if info.data.get("interval") != "SUM":
                raise ValueError("Instruction: the interval must be **SUM** when **time-based grouping is not required**.")
        return v

class DataSoilDashboardQueryPayload(BaseModel):
    model_config = ConfigDict(frozen=False)
    apiCode: str = Field(default="")
    requestId: str = Field(default="")
    applicationCode: str = Field(default="")
    applicationToken: str = Field(default="")
    debug: bool = Field(default=False)
    timeShift: DataSoilDashboardQueryPayloadTimeShift = Field(description="Time comparison analysis config.", default_factory=DataSoilDashboardQueryPayloadTimeShift)
    dynamicQueryParam: DataSoilDashboardQueryPayloadQueryParam
    forceFlush: bool = Field(default=False)

# Resolve forward references
DataSoilDashboardQueryPayload.model_rebuild()

@tool
def query_datasoil_data_tool(payload: DataSoilDashboardQueryPayload) -> str:
    """Queries the DataSoil database with a complex payload."""
    print("--- Tool successfully called with validated payload ---")
    # In a real scenario, you'd process the payload here.
    # For reproduction, we just need to see that it gets called correctly.
    return "Tool call successful."

# Use a model that supports tool calling, like gpt-4o
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Bind the tool to the LLM
llm_with_tools = llm.bind_tools([query_datasoil_data_tool])

# --- NEW: Inspect the schema LangChain generates BEFORE the LLM call ---
tool_schemas = llm_with_tools.kwargs.get("tools", [])
print("\n--- Generated Tool Schema (for LLM) ---")
print(json.dumps(tool_schemas, indent=2))
# --- End of new section ---

# Example invocation
prompt = "Get the detail table for sales data from 2025-07-01 00:00:00 to 2025-07-08 00:00:00, grouped by city, and ordered by total revenue descending."

print(f"\n--- Invoking LLM with prompt: '{prompt}' ---")

ai_msg = llm_with_tools.invoke(prompt)

print("\n--- LLM Response ---")
print(ai_msg)

if isinstance(ai_msg, AIMessage) and ai_msg.tool_calls:
    print("\n--- Generated Tool Call Arguments ---")
    # In a real case, you'd see the arguments the LLM generated.
    # The bug is that these args are often malformed due to an incorrect schema.
    print(ai_msg.tool_calls[0]['args'])
else:
    print("\n--- No tool call was generated ---")
