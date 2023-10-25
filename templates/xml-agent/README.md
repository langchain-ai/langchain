# XML Agent

This template creates an agent that uses XML syntax to communicate its decisions of what actions to take.
For this example, we use Anthropic since Anthropic's Claude models are particularly good at writing XML syntax.
This example creates an agent that can optionally look up things on the internet using You.com's retriever.

##  LLM

This template will use `Anthropic` by default. 

Be sure that `ANTHROPIC_API_KEY` is set in your environment.

##  Tools

This template will use `You.com` by default. 

Be sure that `YDC_API_KEY` is set in your environment.
