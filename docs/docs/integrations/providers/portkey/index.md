# Portkey

>[Portkey](https://docs.portkey.ai/overview/introduction) is a platform designed to streamline the deployment 
> and management of Generative AI applications. 
> It provides comprehensive features for monitoring, managing models,
> and improving the performance of your AI applications.

## LLMOps for Langchain

Portkey brings production readiness to Langchain. With Portkey, you can 
- [x] view detailed **metrics & logs** for all requests, 
- [x] enable **semantic cache** to reduce latency & costs, 
- [x] implement automatic **retries & fallbacks** for failed requests, 
- [x] add **custom tags** to requests for better tracking and analysis and [more](https://docs.portkey.ai).

### Using Portkey with Langchain
Using Portkey is as simple as just choosing which Portkey features you want, enabling them via `headers=Portkey.Config` and passing it in your LLM calls.

To start, get your Portkey API key by [signing up here](https://app.portkey.ai/login). (Click the profile icon on the top left, then click on "Copy API Key")

For OpenAI, a simple integration with logging feature would look like this:
```python
from langchain.llms import OpenAI
from langchain.utilities import Portkey

# Add the Portkey API Key from your account
headers = Portkey.Config(
    api_key = "<PORTKEY_API_KEY>"
)

llm = OpenAI(temperature=0.9, headers=headers)
llm.predict("What would be a good company name for a company that makes colorful socks?")
```
Your logs will be captured on your [Portkey dashboard](https://app.portkey.ai).

A common Portkey X Langchain use case is to **trace a chain or an agent** and view all the LLM calls originating from that request. 

### **Tracing Chains & Agents**

```python
from langchain.agents import AgentType, initialize_agent, load_tools  
from langchain.llms import OpenAI
from langchain.utilities import Portkey

# Add the Portkey API Key from your account
headers = Portkey.Config(
    api_key = "<PORTKEY_API_KEY>",
    trace_id = "fef659"
)

llm = OpenAI(temperature=0, headers=headers)  
tools = load_tools(["serpapi", "llm-math"], llm=llm)  
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)  
  
# Let's test it out!  
agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
```

**You can see the requests' logs along with the trace id on Portkey dashboard:**

<img src="/img/portkey-dashboard.gif" height="250"/>
<img src="/img/portkey-tracing.png" height="250"/>

## Advanced Features

1. **Logging:** Log all your LLM requests automatically by sending them through Portkey. Each request log contains `timestamp`, `model name`, `total cost`, `request time`, `request json`, `response json`, and additional Portkey features.
2. **Tracing:** Trace id can be passed along with each request and is visibe on the logs on Portkey dashboard. You can also set a **distinct trace id** for each request. You can [append user feedback](https://docs.portkey.ai/key-features/feedback-api) to a trace id as well.
3. **Caching:** Respond to previously served customers queries from cache instead of sending them again to OpenAI. Match exact strings OR semantically similar strings. Cache can save costs and reduce latencies by 20x.
4. **Retries:** Automatically reprocess any unsuccessful API requests **`upto 5`** times. Uses an **`exponential backoff`** strategy, which spaces out retry attempts to prevent network overload.
5. **Tagging:** Track and audit each user interaction in high detail with predefined tags.

| Feature | Config Key | Value (Type) | Required/Optional |
| -- | -- | -- | -- |
| API Key | `api_key` | API Key (`string`) | ✅ Required |
| [Tracing Requests](https://docs.portkey.ai/key-features/request-tracing) | `trace_id` | Custom `string` | ❔ Optional |
| [Automatic Retries](https://docs.portkey.ai/key-features/automatic-retries) | `retry_count` | `integer` [1,2,3,4,5] | ❔ Optional |
| [Enabling Cache](https://docs.portkey.ai/key-features/request-caching) | `cache` | `simple` OR `semantic` | ❔ Optional |
| Cache Force Refresh | `cache_force_refresh` | `True` | ❔ Optional |
| Set Cache Expiry | `cache_age` | `integer` (in seconds) | ❔ Optional |
| [Add User](https://docs.portkey.ai/key-features/custom-metadata) | `user` | `string` | ❔ Optional |
| [Add Organisation](https://docs.portkey.ai/key-features/custom-metadata) | `organisation` | `string` | ❔ Optional |
| [Add Environment](https://docs.portkey.ai/key-features/custom-metadata) | `environment` | `string` | ❔ Optional |
| [Add Prompt (version/id/string)](https://docs.portkey.ai/key-features/custom-metadata) | `prompt` | `string` | ❔ Optional |


## **Enabling all Portkey Features:**

```py
headers = Portkey.Config(
    
    # Mandatory
    api_key="<PORTKEY_API_KEY>",  
	
	# Cache Options
    cache="semantic",                 
    cache_force_refresh="True",             
    cache_age=1729,  

    # Advanced
    retry_count=5,                                           
    trace_id="langchain_agent",                          

    # Metadata
    environment="production",        
    user="john",                      
    organisation="acme",             
    prompt="Frost"
    
)
```


For detailed information on each feature and how to use it, [please refer to the Portkey docs](https://docs.portkey.ai). If you have any questions or need further assistance, [reach out to us on Twitter.](https://twitter.com/portkeyai).