"""Test AI21LLM llm."""


from langchain_ai21.llms import AI21LLM


def _generate_llm() -> AI21LLM:
    """
    Testing AI21LLm using non default parameters with the following parameters
    """
    return AI21LLM(
        model="j2-ultra",
        max_tokens=2,  # Use less tokens for a faster response
        temperature=0,  # for a consistent response
        epoch=1,
    )


def test_stream() -> None:
    """Test streaming tokens from AI21."""
    llm = AI21LLM(
        model="j2-ultra",
    )

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_abatch() -> None:
    """Test streaming tokens from AI21LLM."""
    llm = AI21LLM(
        model="j2-ultra",
    )

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from AI21LLM."""
    llm = AI21LLM(
        model="j2-ultra",
    )

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


def test_batch() -> None:
    """Test batch tokens from AI21LLM."""
    llm = AI21LLM(
        model="j2-ultra",
    )

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from AI21LLM."""
    llm = AI21LLM(
        model="j2-ultra",
    )

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)


def test_invoke() -> None:
    """Test invoke tokens from AI21LLM."""
    llm = AI21LLM(
        model="j2-ultra",
    )

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)


def test__generate() -> None:
    llm = _generate_llm()
    llm_result = llm.generate(
        prompts=["Hey there, my name is Pickle Rick. What is your name?"],
        stop=["##"],
    )

    assert len(llm_result.generations) > 0
    assert llm_result.llm_output["token_count"] != 0  # type: ignore


async def test__agenerate() -> None:
    llm = _generate_llm()
    llm_result = await llm.agenerate(
        prompts=["Hey there, my name is Pickle Rick. What is your name?"],
        stop=["##"],
    )

    assert len(llm_result.generations) > 0
    assert llm_result.llm_output["token_count"] != 0  # type: ignore
