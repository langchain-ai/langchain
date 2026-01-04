"""QWED Neurosymbolic Verification Tool for LangChain."""
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool


class QWEDInput(BaseModel):
    """Input schema for QWED verification tool."""
    
    query: str = Field(
        description=(
            "Query to verify. Can be a mathematical expression, "
            "logical statement, or code snippet. Examples: "
            "'What is the derivative of x^2?', "
            "'Is (A AND NOT A) satisfiable?', "
            "'Is eval(user_input) safe?'"
        )
    )


class QWEDTool(BaseTool):
    """Tool for neurosymbolic verification using QWED.
    
    QWED combines neural networks (LLMs) with symbolic reasoning 
    (SymPy, Z3, AST) to provide deterministic verification of AI outputs.
    
    Supports verification of:
    - Mathematical calculations (calculus, algebra, etc.)
    - Logical statements (SAT, propositional logic)
    - Code security (dangerous patterns, eval/exec detection)
    
    Returns verified results with 100% confidence or verification failures.
    
    Setup:
        Install QWED: pip install qwed
        
    Examples:
        >>> from langchain_community.tools import QWEDTool
        >>> tool = QWEDTool(provider="openai", api_key="sk-...")
        >>> result = tool.run("What is the derivative of x^2?")
        >>> print(result)
        ✅ VERIFIED: 2*x
        Confidence: 100%
        Method: symbolic
    """
    
    name: str = "qwed_verify"
    description: str = (
        "Verify mathematical calculations, logical statements, or code "
        "using neurosymbolic AI (LLM + symbolic reasoning). "
        "Returns deterministic proof (100% confidence) or failure. "
        "Use for: math problems, logic puzzles, code security checks. "
        "Input should be a clear question or statement to verify."
    )
    args_schema: Type[BaseModel] = QWEDInput
    
    # QWED configuration
    provider: str = "openai"
    """LLM provider: 'openai', 'anthropic', 'gemini', or None for custom"""
    
    api_key: Optional[str] = None
    """API key for the LLM provider (or set via environment variable)"""
    
    model: str = "gpt-4o-mini"
    """Model name (e.g., 'gpt-4o-mini', 'claude-3-haiku-20240307')"""
    
    mask_pii: bool = False
    """Enable PII masking for privacy protection (requires qwed[pii])"""
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute QWED verification.
        
        Args:
            query: Query to verify
            run_manager: Callback manager (optional)
            
        Returns:
            Formatted verification result string
            
        Raises:
            ImportError: If QWED is not installed
        """
        try:
            from qwed_sdk import QWEDLocal
        except ImportError as e:
            raise ImportError(
                "QWED is not installed. Install with: pip install qwed\n"
                "For PII masking support: pip install 'qwed[pii]'\n"
                "See: https://github.com/QWED-AI/qwed-verification"
            ) from e
        
        # Initialize QWED client
        client = QWEDLocal(
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
            mask_pii=self.mask_pii
        )
        
        # Run verification
        result = client.verify(query)
        
        # Format output
        if result.verified:
            return (
                f"✅ VERIFIED: {result.value}\n"
                f"Confidence: {result.confidence:.0%}\n"
                f"Method: {result.evidence.get('method', 'symbolic')}"
            )
        else:
            return (
                f"❌ VERIFICATION FAILED\n"
                f"Reason: {result.error or 'Could not verify statement'}\n"
                f"Confidence: {result.confidence:.0%}"
            )
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async version - delegates to sync for now.
        
        Args:
            query: Query to verify
            run_manager: Callback manager (optional)
            
        Returns:
            Formatted verification result string
        """
        # QWED doesn't have async support yet, so we delegate to sync
        return self._run(query, run_manager)
