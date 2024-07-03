from langchain_glm.chat_models.all_tools_message import (
    _paser_chunk,
    default_all_tool_chunk_parser,
)


def test_paser_web_browser_invalid_tool_calls():
    raw_tool_calls = [
        {
            "id": "call_CbC42Tnsn4507cig1welo",
            "type": "web_browser",
            "web_browser": {"input": ""},
        }
    ]
    tool_call_chunks = default_all_tool_chunk_parser(raw_tool_calls)

    print(tool_call_chunks)
    assert tool_call_chunks[0]["name"] == "web_browser"
    assert tool_call_chunks[0]["args"] == '{"input": ""}'
    assert tool_call_chunks[0]["id"] == "call_CbC42Tnsn4507cig1welo"
    paser, invalid_tool_calls = _paser_chunk(tool_call_chunks)

    print(paser)
    print(invalid_tool_calls)

    assert invalid_tool_calls[0]["name"] == "web_browser"
    assert invalid_tool_calls[0]["args"] == '{"input": ""}'
    assert invalid_tool_calls[0]["id"] == "call_CbC42Tnsn4507cig1welo"


def test_paser_web_browser_success_tool_calls():
    raw_tool_calls = [
        {
            "type": "web_browser",
            "web_browser": {
                "outputs": [
                    {
                        "title": "昨夜今晨，京津冀发生这些大事（2024年6月27日） - 腾讯网",
                        "link": "https://new.qq.com/rain/a/20240627A013AI00",
                        "content": "网页1 天前 · 昨夜今晨，京津冀发生这些大事（2024年6月27日）. 北京首套房首付比例最低2成. “517”楼市新政的“靴子”在北京落地了。. 昨天，北京市住建委、中国人民银行北京市分行 …   ",
                    }
                ]
            },
        }
    ]
    tool_call_chunks = default_all_tool_chunk_parser(raw_tool_calls)

    print(tool_call_chunks)
    assert tool_call_chunks[0]["name"] == "web_browser"
    assert (
        tool_call_chunks[0]["args"]
        == '{"outputs": [{"title": "昨夜今晨，京津冀发生这些大事（2024年6月27日） - 腾讯网", "link": "https://new.qq.com/rain/a/20240627A013AI00", "content": "网页1 天前\u2002·\u2002昨夜今晨，京津冀发生这些大事（2024年6月27日）. 北京首套房首付比例最低2成. “517”楼市新政的“靴子”在北京落地了。. 昨天，北京市住建委、中国人民银行北京市分行 … \xa0 "}]}'
    )
    paser, invalid_tool_calls = _paser_chunk(tool_call_chunks)

    print(paser)
    print(invalid_tool_calls)

    assert paser[0]["name"] == "web_browser"
    assert paser[0]["args"] == {
        "outputs": [
            {
                "title": "昨夜今晨，京津冀发生这些大事（2024年6月27日） - 腾讯网",
                "link": "https://new.qq.com/rain/a/20240627A013AI00",
                "content": "网页1 天前\u2002·\u2002昨夜今晨，京津冀发生这些大事（2024年6月27日）. 北京首套房首付比例最低2成. “517”楼市新政的“靴子”在北京落地了。. 昨天，北京市住建委、中国人民银行北京市分行 … \xa0 ",
            }
        ]
    }


def test_paser_code_interpreter_invalid_tool_calls():
    raw_tool_calls = [
        {
            "id": "call_zp-asyyxPwwn_W9POu8bK",
            "type": "code_interpreter",
            "code_interpreter": {"input": "x"},
        }
    ]
    tool_call_chunks = default_all_tool_chunk_parser(raw_tool_calls)

    print(tool_call_chunks)
    assert tool_call_chunks[0]["name"] == "code_interpreter"
    assert tool_call_chunks[0]["args"] == '{"input": "x"}'
    assert tool_call_chunks[0]["id"] == "call_zp-asyyxPwwn_W9POu8bK"
    paser, invalid_tool_calls = _paser_chunk(tool_call_chunks)

    print(paser)
    print(invalid_tool_calls)

    assert invalid_tool_calls[0]["name"] == "code_interpreter"
    assert invalid_tool_calls[0]["args"] == '{"input": "x"}'
    assert invalid_tool_calls[0]["id"] == "call_zp-asyyxPwwn_W9POu8bK"


def test_paser_code_interpreter_success_tool_calls():
    raw_tool_calls = [
        {"type": "code_interpreter", "code_interpreter": {"outputs": [{"log": "100"}]}}
    ]
    tool_call_chunks = default_all_tool_chunk_parser(raw_tool_calls)

    print(tool_call_chunks)
    assert tool_call_chunks[0]["name"] == "code_interpreter"
    assert tool_call_chunks[0]["args"] == '{"outputs": [{"log": "100"}]}'
    paser, invalid_tool_calls = _paser_chunk(tool_call_chunks)

    print(paser)
    print(invalid_tool_calls)

    assert paser[0]["name"] == "code_interpreter"
    assert paser[0]["args"] == {"outputs": [{"log": "100"}]}


def test_paser_drawing_tool_invalid_tool_calls():
    raw_tool_calls = [
        {
            "id": "call_zp-asyyxPwwn_W9POu8bK",
            "type": "drawing_tool",
            "drawing_tool": {"input": "x"},
        }
    ]
    tool_call_chunks = default_all_tool_chunk_parser(raw_tool_calls)

    print(tool_call_chunks)
    assert tool_call_chunks[0]["name"] == "drawing_tool"
    assert tool_call_chunks[0]["args"] == '{"input": "x"}'
    assert tool_call_chunks[0]["id"] == "call_zp-asyyxPwwn_W9POu8bK"
    paser, invalid_tool_calls = _paser_chunk(tool_call_chunks)

    print(paser)
    print(invalid_tool_calls)

    assert invalid_tool_calls[0]["name"] == "drawing_tool"
    assert invalid_tool_calls[0]["args"] == '{"input": "x"}'
    assert invalid_tool_calls[0]["id"] == "call_zp-asyyxPwwn_W9POu8bK"


def test_paser_drawing_tool_success_tool_calls():
    raw_tool_calls = [
        {"type": "drawing_tool", "drawing_tool": {"outputs": [{"image": "http://"}]}}
    ]
    tool_call_chunks = default_all_tool_chunk_parser(raw_tool_calls)

    print(tool_call_chunks)
    assert tool_call_chunks[0]["name"] == "drawing_tool"
    assert tool_call_chunks[0]["args"] == '{"outputs": [{"image": "http://"}]}'
    paser, invalid_tool_calls = _paser_chunk(tool_call_chunks)

    print(paser)
    print(invalid_tool_calls)

    assert paser[0]["name"] == "drawing_tool"
    assert paser[0]["args"] == {"outputs": [{"image": "http://"}]}


def test_paser_function_invalid_tool_calls():
    raw_tool_calls = [
        {
            "id": "call_Tp4cX0Qh1S37un60DUkH8",
            "type": "function",
            "function": {"name": "get_current_weather", "arguments": ""},
        }
    ]
    tool_call_chunks = default_all_tool_chunk_parser(raw_tool_calls)

    print(tool_call_chunks)
    assert tool_call_chunks[0]["name"] == "get_current_weather"
    assert tool_call_chunks[0]["args"] == ""
    assert tool_call_chunks[0]["id"] == "call_Tp4cX0Qh1S37un60DUkH8"
    paser, invalid_tool_calls = _paser_chunk(tool_call_chunks)

    print(paser)
    print(invalid_tool_calls)

    assert invalid_tool_calls[0]["name"] == "get_current_weather"
    assert invalid_tool_calls[0]["args"] == ""
    assert invalid_tool_calls[0]["id"] == "call_Tp4cX0Qh1S37un60DUkH8"


def test_paser_function_success_tool_calls():
    raw_tool_calls = [
        {
            "id": "call_Tp4cX0Qh1S37un60DUkH8",
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "arguments": '{"location":"北京"," unit":"fahrenheit"}',
            },
        }
    ]
    tool_call_chunks = default_all_tool_chunk_parser(raw_tool_calls)

    print(tool_call_chunks)
    assert tool_call_chunks[0]["name"] == "get_current_weather"
    assert tool_call_chunks[0]["args"] == '{"location":"北京"," unit":"fahrenheit"}'
    assert tool_call_chunks[0]["id"] == "call_Tp4cX0Qh1S37un60DUkH8"
    paser, invalid_tool_calls = _paser_chunk(tool_call_chunks)

    print(paser)
    print(invalid_tool_calls)

    assert paser[0]["name"] == "get_current_weather"
    assert paser[0]["args"] == {"location": "北京", "unit": "fahrenheit"}
    assert paser[0]["id"] == "call_Tp4cX0Qh1S37un60DUkH8"
