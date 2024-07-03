def test_tit():
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    print(enc.encode("hello world"))
    print(enc.decode([15339, 1917]))

    assert enc.decode(enc.encode("hello world")) == "hello world"
