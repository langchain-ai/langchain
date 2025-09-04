"""
Advanced usage examples for LangChain Llama Stack integration.

This example demonstrates more complex scenarios and integrations.
"""


from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_llamastack import (
    LlamaStackSafety,
    create_llamastack_llm,
    get_llamastack_models,
)


def advanced_prompt_example():
    """Demonstrate advanced prompting with LangChain templates."""
    print("📝 Advanced Prompting Example")
    print("=" * 40)

    try:
        llm = create_llamastack_llm(model="ollama/llama3:70b-instruct")

        # Create a detailed prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are an expert AI researcher \
                    who explains concepts clearly and concisely."
                ),
                HumanMessage(
                    content="""Explain the concept of \
                    {topic} in the context of {context}.

            Please structure your response as follows:
            1. Brief definition
            2. Key components
            3. Real-world applications
            4. Future implications

            Keep the explanation accessible to {audience}."""
                ),
            ]
        )

        # Create a chain
        chain = LLMChain(llm=llm, prompt=prompt)

        # Test with different topics
        examples = [
            {
                "topic": "neural networks",
                "context": "artificial intelligence",
                "audience": "undergraduate students",
            },
            {
                "topic": "large language models",
                "context": "natural language processing",
                "audience": "technical professionals",
            },
        ]

        for example in examples:
            print(f"\nTopic: {example['topic']} for {example['audience']}")
            print("-" * 30)

            result = chain.run(**example)
            print(result[:300] + "..." if len(result) > 300 else result)

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def conversation_memory_example():
    """Demonstrate conversation with memory using LangChain."""
    print("\n💭 Conversation with Memory Example")
    print("=" * 40)

    try:
        llm = create_llamastack_llm(model="ollama/llama3:70b-instruct")

        # Create conversation chain with memory
        memory = ConversationBufferMemory()
        conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

        # Simulate a conversation
        turns = [
            "Hi, I'm learning about machine learning. Can you help?",
            "What's the difference between supervised and unsupervised learning?",
            "Can you give me examples of supervised learning algorithms?",
            "What about the algorithm you just mentioned - linear regression?",
        ]

        for turn in turns:
            print(f"\nHuman: {turn}")
            response = conversation.predict(input=turn)
            print(
                f"AI: {response[:200]}..." if len(response) > 200 else f"AI: {response}"
            )

        # Show conversation history
        print("\nConversation History:")
        print(f"Memory buffer: {len(memory.buffer.split())} words")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def safe_conversation_agent():
    """Create a safety-aware conversational agent."""
    print("\n🛡️ Safe Conversation Agent Example")
    print("=" * 40)

    try:
        llm = create_llamastack_llm(model="ollama/llama3:70b-instruct")

        safety = LlamaStackSafety(
            base_url="http://localhost:8321", shield_id="code-scanner"
        )

        class SafeConversationalAgent:
            def __init__(self, llm, safety_checker):
                self.llm = llm
                self.safety = safety_checker
                self.memory = ConversationBufferMemory()
                self.conversation = ConversationChain(
                    llm=llm, memory=self.memory, verbose=False
                )

                # Set system context
                self.conversation.predict(
                    input="You are a helpful, harmless, and honest AI assistant. "
                    "Always provide accurate information and\
                     decline inappropriate requests politely."
                )

            def chat(self, user_input: str) -> dict:
                """Safe chat with comprehensive safety checking."""

                # Check input safety
                input_safety = self.safety.check_content(user_input)
                if not input_safety.is_safe:
                    return {
                        "response": "I can't process that \
                        request due to safety concerns.",
                        "status": "input_rejected",
                        "safety_info": input_safety.to_dict(),
                    }

                try:
                    # Generate response
                    response = self.conversation.predict(input=user_input)

                    # Check output safety
                    output_safety = self.safety.check_content(response)
                    if not output_safety.is_safe:
                        return {
                            "response": "I need to revise my response\
                             for safety reasons. Could you rephrase your question?",
                            "status": "output_filtered",
                            "safety_info": output_safety.to_dict(),
                        }

                    return {
                        "response": response,
                        "status": "success",
                        "safety_info": {"input_safe": True, "output_safe": True},
                    }

                except Exception as e:
                    return {
                        "response": "I encountered an error processing your request.",
                        "status": "error",
                        "error": str(e),
                    }

            def get_conversation_summary(self) -> dict:
                """Get a summary of the conversation."""
                return {
                    "total_exchanges": len(self.memory.buffer.split("Human:")) - 1,
                    "memory_length": len(self.memory.buffer),
                    "last_interactions": (
                        self.memory.buffer.split("Human:")[-3:]
                        if self.memory.buffer
                        else []
                    ),
                }

        # Test the safe agent
        agent = SafeConversationalAgent(llm, safety)

        test_conversations = [
            "Hello! I'm interested in learning about AI ethics.",
            "What are some important considerations when developing AI systems?",
            "How can we ensure AI systems are fair and unbiased?",
            "Can you give me some resources to learn more about this topic?",
        ]

        for msg in test_conversations:
            print(f"\nUser: {msg}")
            result = agent.chat(msg)
            print(f"Status: {result['status']}")
            print(
                f"AI: {result['response'][:150]}..."
                if len(result["response"]) > 150
                else f"AI: {result['response']}"
            )

            if result["status"] != "success":
                print(f"Safety Info: {result.get('safety_info', {})}")

        # Show conversation summary
        summary = agent.get_conversation_summary()
        print("\nConversation Summary:")
        print(f"Total exchanges: {summary['total_exchanges']}")
        print(f"Memory size: {summary['memory_length']} characters")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def multi_model_comparison():
    """Compare responses from different models."""
    print("\n🔄 Multi-Model Comparison Example")
    print("=" * 40)

    try:
        # Get available models using the new approach
        available_models = get_llamastack_models()
        print(f"Available models: {len(available_models)}")

        # Select models for comparison (use first 2 if available)
        test_models = (
            available_models[:2] if len(available_models) >= 2 else available_models
        )

        if len(test_models) < 2:
            print("Need at least 2 models for comparison")
            return False

        # Test prompt
        test_prompt = "Explain the concept of machine learning in exactly 3 sentences."

        print(f"\nPrompt: {test_prompt}")
        print("\nModel Responses:")

        for model in test_models:
            print(f"\n--- {model.split('/')[-1]} ---")

            llm = create_llamastack_llm(model=model)

            try:
                response = llm.invoke(test_prompt)
                print(response.content)

                # Get model info
                model_info = llm.get_model_info()
                if "provider_id" in model_info:
                    print(f"Provider: {model_info['provider_id']}")

            except Exception as e:
                print(f"Error with {model}: {e}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def safety_policy_testing():
    """Test different safety policies and shields."""
    print("\n🛡️ Safety Policy Testing Example")
    print("=" * 40)

    try:
        safety = LlamaStackSafety(
            base_url="http://localhost:8321", shield_id="code-scanner"
        )

        # Get available shields
        shields = safety.get_available_shields()
        print(f"Available shields: {shields}")

        if not shields:
            print("No safety shields available for testing")
            return False

        # Test content with different safety levels
        test_contents = [
            ("Harmless greeting", "Hello, how are you today?"),
            ("Educational request", "Can you explain quantum computing?"),
            ("Help request", "I need help with my homework assignment."),
            ("Technical question", "What are the best practices for AI development?"),
        ]

        # Test with different shields (if multiple available)
        shields_to_test = shields[:2]  # Test first 2 shields

        for shield_id in shields_to_test:
            print(f"\n--- Testing with {shield_id} ---")

            # Get shield info
            shield_info = safety.get_shield_info(shield_id)
            print(f"Shield info: {shield_info}")

            for label, content in test_contents:
                result = safety.check_content(content, shield_id=shield_id)
                status = "✅ SAFE" if result.is_safe else "❌ UNSAFE"
                print(f"{status} - {label}: {content[:50]}...")

                if not result.is_safe:
                    print(f"  Violation: {result.violation_type}")
                    print(f"  Confidence: {result.confidence_score:.2f}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def langchain_integration_showcase():
    """Showcase integration with various LangChain components."""
    print("\n🔗 LangChain Integration Showcase")
    print("=" * 40)

    try:
        llm = create_llamastack_llm(model="ollama/llama3:70b-instruct")

        # 1. Simple Chain
        print("\n1. Simple LLM Chain:")
        simple_prompt = PromptTemplate(
            input_variables=["topic"],
            template="Write a brief, engaging introduction to {topic}.",
        )
        simple_chain = LLMChain(llm=llm, prompt=simple_prompt)
        result = simple_chain.run(topic="artificial intelligence")
        print(result[:200] + "..." if len(result) > 200 else result)

        # 2. Chat with System Message
        print("\n2. Chat with System Message:")
        messages = [
            SystemMessage(
                content="You are a helpful coding assistant. \
                Provide concise, practical advice."
            ),
            HumanMessage(
                content="How do I debug a Python script that's running slowly?"
            ),
        ]
        response = llm.invoke(messages)
        print(
            response.content[:200] + "..."
            if len(response.content) > 200
            else response.content
        )

        # 3. Streaming Example
        print("\n3. Streaming Response:")
        print("AI: ", end="", flush=True)
        stream_count = 0
        for chunk in llm.stream(
            "Explain the benefits of using LangChain in 2 sentences."
        ):
            print(chunk.content, end="", flush=True)
            stream_count += 1
            if stream_count > 20:  # Limit for demo
                print("...")
                break
        print()

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all advanced examples."""
    print("🚀 LangChain Llama Stack Integration - Advanced Examples")
    print("=" * 70)

    examples = [
        ("Advanced Prompting", advanced_prompt_example),
        ("Conversation Memory", conversation_memory_example),
        ("Safe Conversation Agent", safe_conversation_agent),
        ("Multi-Model Comparison", multi_model_comparison),
        ("Safety Policy Testing", safety_policy_testing),
        ("LangChain Integration", langchain_integration_showcase),
    ]

    results = []

    for name, example_func in examples:
        try:
            print(f"\n{'='*70}")
            print(f"Running: {name}")
            print("=" * 70)

            success = example_func()
            results.append((name, success))

            if success:
                print(f"✅ {name} completed successfully")
            else:
                print(f"❌ {name} failed")

        except Exception as e:
            print(f"❌ {name} error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("📊 Results Summary")
    print("=" * 70)

    successful = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")

    print(f"\nOverall: {successful}/{total} examples completed successfully")

    if successful == total:
        print("🎉 All advanced examples passed!")
        print("\n💡 The LangChain Llama Stack integration is working perfectly!")
        print("You can now use it for:")
        print("- Complex conversational AI systems")
        print("- Safety-aware AI applications")
        print("- Multi-model AI workflows")
        print("- Advanced prompt engineering")
        print("- Memory-enabled chat systems")
    else:
        print("\n⚠️ Some examples failed. This might be due to:")
        print("- Missing models in your Llama Stack setup")
        print("- Safety shields not configured")
        print("- Server connectivity issues")
        print("- Model-specific limitations")


if __name__ == "__main__":
    main()
