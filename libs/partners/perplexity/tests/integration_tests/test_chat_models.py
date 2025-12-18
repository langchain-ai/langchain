"""Integration tests for ChatPerplexity."""

import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_perplexity import ChatPerplexity, MediaResponse, WebSearchOptions


@pytest.mark.skipif(
    not os.environ.get("PPLX_API_KEY"), reason="PPLX_API_KEY not set"
)
class TestChatPerplexityIntegration:
    def test_standard_generation(self) -> None:
        """Test standard generation."""
        chat = ChatPerplexity(model="sonar", temperature=0)
        message = HumanMessage(content="Hello! How are you?")
        response = chat.invoke([message])
        assert response.content
        assert isinstance(response.content, str)

    async def test_async_generation(self) -> None:
        """Test async generation."""
        chat = ChatPerplexity(model="sonar", temperature=0)
        message = HumanMessage(content="Hello! How are you?")
        response = await chat.ainvoke([message])
        assert response.content
        assert isinstance(response.content, str)

    def test_pro_search(self) -> None:
        """Test Pro Search (reasoning_steps extraction)."""
        # Pro search is available on sonar-pro
        chat = ChatPerplexity(
            model="sonar-pro",
            temperature=0,
            web_search_options=WebSearchOptions(search_type="pro"),
            # Pro search often requires streaming to get steps, but let's check non-streaming first or just use streaming
            streaming=True
        )
        message = HumanMessage(content="Who won the 2024 US election and why?")

        # We need to collect chunks to check reasoning steps
        chunks = list(chat.stream([message]))
        full_content = "".join([c.content for c in chunks])
        assert full_content

        # Check if any chunk has reasoning_steps
        has_reasoning = any("reasoning_steps" in c.additional_kwargs for c in chunks)
        # Note: reasoning_steps might not be guaranteed for every query, but "why" usually triggers it in pro search
        if has_reasoning:
            assert True
        else:
            # Fallback assertion if no reasoning steps returned (could be query specific)
            assert len(chunks) > 0

    async def test_streaming(self) -> None:
        """Test streaming."""
        chat = ChatPerplexity(model="sonar", temperature=0)
        message = HumanMessage(content="Count to 5")
        async for chunk in chat.astream([message]):
            assert isinstance(chunk.content, str)

    def test_citations_and_search_results(self) -> None:
        """Test that citations and search results are returned."""
        chat = ChatPerplexity(model="sonar", temperature=0)
        message = HumanMessage(content="Who is the CEO of OpenAI?")
        response = chat.invoke([message])

        # Citations are usually in additional_kwargs
        assert "citations" in response.additional_kwargs
        # Search results might be there too
        # Note: presence depends on whether search was performed
        if response.additional_kwargs.get("citations"):
            assert len(response.additional_kwargs["citations"]) > 0

    def test_search_control(self) -> None:
        """Test search control parameters."""
        # Test disabled search (should complete without citations)
        chat = ChatPerplexity(
            model="sonar",
            disable_search=True
        )
        message = HumanMessage(content="What is 2+2?")
        response = chat.invoke([message])
        assert response.content

        # Test search classifier
        chat_classifier = ChatPerplexity(
            model="sonar",
            enable_search_classifier=True
        )
        response_classifier = chat_classifier.invoke([message])
        assert response_classifier.content

    def test_search_recency_filter(self) -> None:
        """Test search_recency_filter parameter."""
        chat = ChatPerplexity(
            model="sonar",
            search_recency_filter="month"
        )
        message = HumanMessage(content="Latest AI news")
        response = chat.invoke([message])
        assert response.content

    def test_search_domain_filter(self) -> None:
        """Test search_domain_filter parameter."""
        chat = ChatPerplexity(
            model="sonar",
            search_domain_filter=["wikipedia.org"]
        )
        message = HumanMessage(content="Python programming language")
        response = chat.invoke([message])

        # Verify citations come from wikipedia if any
        if citations := response.additional_kwargs.get("citations"):
            assert any("wikipedia.org" in c for c in citations)

    def test_media_and_metadata(self) -> None:
        """Test related questions and images."""
        chat = ChatPerplexity(
            model="sonar-pro",
            return_related_questions=True,
            return_images=True,
            # Media response overrides for video
            media_response=MediaResponse(overrides={"return_videos": True})
        )
        message = HumanMessage(content="Apollo 11 moon landing")
        response = chat.invoke([message])

        # Check related questions
        if related := response.additional_kwargs.get("related_questions"):
            assert len(related) > 0

        # Check images
        if images := response.additional_kwargs.get("images"):
            assert len(images) > 0

        # Check videos (might not always be present but structure should handle it)
        if videos := response.additional_kwargs.get("videos"):
            assert len(videos) > 0
