"""
SOVR LangChain集成
官方集成 - 准备提交PR到LangChain仓库
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime


# LangChain Callback Handler
class SovrCallbackHandler:
    """
    SOVR Callback Handler for LangChain
    
    Integrates SOVR's AI responsibility layer into LangChain agents.
    Every tool call, LLM invocation, and chain execution is checked
    against SOVR's safety rules.
    
    Usage:
        from langchain.agents import AgentExecutor
        from sovr.integrations.langchain import SovrCallbackHandler
        
        sovr_handler = SovrCallbackHandler(api_key="...")
        
        agent = AgentExecutor(
            agent=...,
            tools=...,
            callbacks=[sovr_handler]
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        block_on_violation: bool = True,
        log_all_actions: bool = True,
        industry: Optional[str] = None,
    ):
        self.api_key = api_key
        self.block_on_violation = block_on_violation
        self.log_all_actions = log_all_actions
        self.industry = industry
        
        # 延迟导入sovr客户端
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from sovr import Sovr
                self._client = Sovr(
                    api_key=self.api_key,
                    industry=self.industry,
                )
            except ImportError:
                raise ImportError("sovr package required. Install with: pip install sovr")
        return self._client
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""
        if self.log_all_actions:
            self.client.log(
                action="llm_start",
                outcome="started",
                context={
                    "model": serialized.get("name", "unknown"),
                    "prompt_count": len(prompts),
                },
            )
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        if self.log_all_actions:
            self.client.log(
                action="llm_end",
                outcome="success",
                context={
                    "generation_count": len(response.generations) if response.generations else 0,
                },
            )
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM errors."""
        self.client.log(
            action="llm_error",
            outcome="error",
            context={"error": str(error)[:500]},
        )
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts running - THIS IS THE KEY INTEGRATION POINT."""
        tool_name = serialized.get("name", "unknown_tool")
        
        # 检查工具调用
        result = self.client.check(
            action=f"tool_{tool_name}",
            context={
                "tool_name": tool_name,
                "input": input_str[:1000],  # 截断
                "tool_description": serialized.get("description", "")[:200],
            },
            agent_id=kwargs.get("run_id"),
        )
        
        if not result.allowed and self.block_on_violation:
            raise ToolBlockedError(
                f"SOVR blocked tool '{tool_name}': {result.reason}",
                tool_name=tool_name,
                rule_id=result.rule_id,
            )
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when tool ends running."""
        if self.log_all_actions:
            self.client.log(
                action="tool_end",
                outcome="success",
                context={"output_length": len(output)},
            )
    
    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when tool errors."""
        self.client.log(
            action="tool_error",
            outcome="error",
            context={"error": str(error)[:500]},
        )
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when chain starts running."""
        chain_name = serialized.get("name", "unknown_chain")
        
        result = self.client.check(
            action=f"chain_{chain_name}",
            context={
                "chain_name": chain_name,
                "input_keys": list(inputs.keys()),
            },
        )
        
        if not result.allowed and self.block_on_violation:
            raise ChainBlockedError(
                f"SOVR blocked chain '{chain_name}': {result.reason}",
                chain_name=chain_name,
            )
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain ends running."""
        if self.log_all_actions:
            self.client.log(
                action="chain_end",
                outcome="success",
                context={"output_keys": list(outputs.keys())},
            )
    
    def on_agent_action(self, action, **kwargs: Any) -> None:
        """Called when agent takes an action."""
        result = self.client.check(
            action="agent_action",
            context={
                "tool": action.tool,
                "tool_input": str(action.tool_input)[:500],
                "log": action.log[:200] if action.log else None,
            },
        )
        
        if not result.allowed and self.block_on_violation:
            raise AgentActionBlockedError(
                f"SOVR blocked agent action '{action.tool}': {result.reason}",
                tool=action.tool,
            )
    
    def on_agent_finish(self, finish, **kwargs: Any) -> None:
        """Called when agent finishes."""
        self.client.log(
            action="agent_finish",
            outcome="success",
            context={
                "return_values": list(finish.return_values.keys()),
            },
        )


class ToolBlockedError(Exception):
    """Tool execution blocked by SOVR."""
    def __init__(self, message: str, tool_name: str, rule_id: Optional[str] = None):
        self.tool_name = tool_name
        self.rule_id = rule_id
        super().__init__(message)


class ChainBlockedError(Exception):
    """Chain execution blocked by SOVR."""
    def __init__(self, message: str, chain_name: str):
        self.chain_name = chain_name
        super().__init__(message)


class AgentActionBlockedError(Exception):
    """Agent action blocked by SOVR."""
    def __init__(self, message: str, tool: str):
        self.tool = tool
        super().__init__(message)


# Tool Wrapper
def sovr_tool(func=None, *, action: Optional[str] = None, api_key: Optional[str] = None):
    """
    Decorator to wrap LangChain tools with SOVR protection.
    
    Usage:
        from langchain.tools import tool
        from sovr.integrations.langchain import sovr_tool
        
        @tool
        @sovr_tool(action="file_operations")
        def delete_file(path: str) -> str:
            '''Delete a file'''
            os.remove(path)
            return f"Deleted {path}"
    """
    def decorator(fn):
        import functools
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                from sovr import Sovr
                client = Sovr(api_key=api_key)
            except ImportError:
                # SOVR not installed, pass through
                return fn(*args, **kwargs)
            
            tool_action = action or f"tool_{fn.__name__}"
            result = client.check(
                action=tool_action,
                context={
                    "function": fn.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200],
                },
            )
            
            if not result.allowed:
                raise ToolBlockedError(
                    f"Tool '{fn.__name__}' blocked: {result.reason}",
                    tool_name=fn.__name__,
                    rule_id=result.rule_id,
                )
            
            return fn(*args, **kwargs)
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


# Agent Safety Wrapper
class SovrAgentExecutor:
    """
    SOVR-protected AgentExecutor wrapper.
    
    Usage:
        from langchain.agents import AgentExecutor
        from sovr.integrations.langchain import SovrAgentExecutor
        
        base_agent = AgentExecutor(agent=..., tools=...)
        safe_agent = SovrAgentExecutor(base_agent, api_key="...")
        
        result = safe_agent.invoke({"input": "..."})
    """
    
    def __init__(
        self,
        agent_executor,
        api_key: Optional[str] = None,
        max_iterations: int = 10,
        block_dangerous_tools: bool = True,
    ):
        self.agent_executor = agent_executor
        self.api_key = api_key
        self.max_iterations = max_iterations
        self.block_dangerous_tools = block_dangerous_tools
        
        # 注入SOVR callback
        sovr_callback = SovrCallbackHandler(
            api_key=api_key,
            block_on_violation=block_dangerous_tools,
        )
        
        if hasattr(agent_executor, 'callbacks'):
            if agent_executor.callbacks is None:
                agent_executor.callbacks = []
            agent_executor.callbacks.append(sovr_callback)
    
    def invoke(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute agent with SOVR protection."""
        return self.agent_executor.invoke(inputs, **kwargs)
    
    async def ainvoke(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Async execute agent with SOVR protection."""
        return await self.agent_executor.ainvoke(inputs, **kwargs)


if __name__ == "__main__":
    print("SOVR LangChain Integration")
    print("=" * 50)
    print("""
Usage example:

    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain_openai import ChatOpenAI
    from sovr.integrations.langchain import SovrCallbackHandler

    # Create SOVR callback
    sovr_handler = SovrCallbackHandler(
        api_key="your-sovr-api-key",
        industry="fintech",
        block_on_violation=True
    )

    # Create agent with SOVR protection
    llm = ChatOpenAI()
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[sovr_handler]
    )

    # All tool calls are now checked by SOVR
    result = agent_executor.invoke({"input": "Delete all files"})
    # -> Blocked by SOVR if dangerous
""")
