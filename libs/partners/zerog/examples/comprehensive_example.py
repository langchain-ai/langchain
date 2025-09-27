"""Comprehensive example of 0G Compute Network integration with LangChain."""

import asyncio
import os
from typing import Any, Dict

from pydantic import BaseModel, Field

from langchain_zerog import ChatZeroG


class WeatherInfo(BaseModel):
    """Weather information structure."""
    location: str = Field(description="The location")
    temperature: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Weather condition")
    humidity: int = Field(description="Humidity percentage")


class GetWeather(BaseModel):
    """Get weather information for a location."""
    location: str = Field(description="City and state/country, e.g. 'San Francisco, CA'")
    unit: str = Field(default="celsius", description="Temperature unit (celsius or fahrenheit)")


async def main() -> None:
    """Run comprehensive 0G integration examples."""
    # Initialize the model
    llm = ChatZeroG(
        model="llama-3.3-70b-instruct",
        temperature=0.7,
        max_tokens=1000,
        # private_key will be read from ZEROG_PRIVATE_KEY environment variable
    )

    print("🚀 0G Compute Network - Comprehensive Integration Example")
    print("=" * 60)

    # 1. Account Management
    print("\n1️⃣ Account Management")
    try:
        # Check initial balance
        balance = await llm.get_balance()
        print(f"   💰 Current balance: {balance['available']} OG tokens")

        # Fund account if needed (uncomment to actually fund)
        # await llm.fund_account("0.1")
        # print("   ✅ Account funded with 0.1 OG tokens")

    except Exception as e:
        print(f"   ⚠️  Account management: {e}")

    # 2. Basic Chat
    print("\n2️⃣ Basic Chat Interaction")
    try:
        messages = [
            ("system", "You are a helpful AI assistant specialized in technology."),
            ("human", "Explain quantum computing in simple terms."),
        ]

        response = await llm.ainvoke(messages)
        print(f"   🤖 Response: {response.content[:200]}...")
        print(f"   📊 Usage: {response.usage_metadata}")
        print(f"   🔗 Provider: {response.response_metadata.get('provider_address', 'N/A')}")

    except Exception as e:
        print(f"   ❌ Basic chat failed: {e}")

    # 3. Streaming Response
    print("\n3️⃣ Streaming Response")
    try:
        print("   🌊 Streaming response: ", end="")

        stream_messages = [
            ("human", "Write a short poem about artificial intelligence.")
        ]

        async for chunk in llm.astream(stream_messages):
            print(chunk.content, end="", flush=True)

        print("\n   ✅ Streaming completed")

    except Exception as e:
        print(f"\n   ❌ Streaming failed: {e}")

    # 4. Tool Calling
    print("\n4️⃣ Tool Calling")
    try:
        # Bind tools to the model
        llm_with_tools = llm.bind_tools([GetWeather])

        tool_messages = [
            ("human", "What's the weather like in Tokyo, Japan?")
        ]

        tool_response = await llm_with_tools.ainvoke(tool_messages)

        if tool_response.tool_calls:
            print(f"   🔧 Tool called: {tool_response.tool_calls[0]['name']}")
            print(f"   📋 Arguments: {tool_response.tool_calls[0]['args']}")
        else:
            print(f"   💬 Response: {tool_response.content}")

    except Exception as e:
        print(f"   ❌ Tool calling failed: {e}")

    # 5. Structured Output
    print("\n5️⃣ Structured Output")
    try:
        structured_llm = llm.with_structured_output(WeatherInfo)

        weather_query = [
            ("human", "Generate weather information for Paris, France. "
                     "Temperature should be 18 degrees, partly cloudy, 65% humidity.")
        ]

        weather_info = await structured_llm.ainvoke(weather_query)

        print(f"   🌍 Location: {weather_info.location}")
        print(f"   🌡️  Temperature: {weather_info.temperature}°C")
        print(f"   ☁️  Condition: {weather_info.condition}")
        print(f"   💧 Humidity: {weather_info.humidity}%")

    except Exception as e:
        print(f"   ❌ Structured output failed: {e}")

    # 6. Model Comparison
    print("\n6️⃣ Model Comparison")
    models_to_test = ["llama-3.3-70b-instruct", "deepseek-r1-70b"]

    for model_name in models_to_test:
        try:
            model_llm = ChatZeroG(
                model=model_name,
                temperature=0.7,
                max_tokens=100,
            )

            test_message = [("human", "What is the meaning of life?")]
            response = await model_llm.ainvoke(test_message)

            print(f"   🤖 {model_name}: {response.content[:100]}...")

        except Exception as e:
            print(f"   ❌ {model_name} failed: {e}")

    # 7. Advanced Features
    print("\n7️⃣ Advanced Configuration")
    try:
        advanced_llm = ChatZeroG(
            model="llama-3.3-70b-instruct",
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
        )

        advanced_messages = [
            ("system", "You are a precise technical writer."),
            ("human", "Explain the benefits of decentralized AI inference."),
        ]

        response = await advanced_llm.ainvoke(advanced_messages)
        print(f"   📝 Technical response: {response.content[:150]}...")

    except Exception as e:
        print(f"   ❌ Advanced configuration failed: {e}")

    print("\n🎉 Comprehensive example completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Account management (balance, funding)")
    print("   ✅ Basic chat interactions")
    print("   ✅ Real-time streaming responses")
    print("   ✅ Tool calling with function schemas")
    print("   ✅ Structured output with Pydantic models")
    print("   ✅ Multiple model support")
    print("   ✅ Advanced parameter configuration")
    print("\n🔗 Learn more: https://docs.0g.ai/")


if __name__ == "__main__":
    # Ensure you have set ZEROG_PRIVATE_KEY environment variable
    if not os.getenv("ZEROG_PRIVATE_KEY"):
        print("❌ Please set ZEROG_PRIVATE_KEY environment variable")
        print("   export ZEROG_PRIVATE_KEY='your-ethereum-private-key'")
        exit(1)

    asyncio.run(main())
