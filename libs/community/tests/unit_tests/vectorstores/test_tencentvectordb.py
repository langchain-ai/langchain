import importlib.util

from langchain_community.vectorstores.tencentvectordb import translate_filter


def test_translate_filter() -> None:
    raw_filter = (
        'and(or(eq("artist", "Taylor Swift"), '
        'eq("artist", "Katy Perry")), lt("length", 180))'
    )
    spec = importlib.util.find_spec("langchain.chains.query_constructor.base")
    if spec is None:
        try:
            translate_filter(raw_filter)
        except ModuleNotFoundError:
            pass
        else:
            assert False
    else:
        result = translate_filter(raw_filter)
        expr = '(artist = "Taylor Swift" or artist = "Katy Perry") ' "and length < 180"
        assert expr == result


def test_translate_filter_with_in_comparison() -> None:
    raw_filter = 'in("artist", ["Taylor Swift", "Katy Perry"])'
    spec = importlib.util.find_spec("langchain.chains.query_constructor.base")
    if spec is None:
        try:
            translate_filter(raw_filter)
        except ModuleNotFoundError:
            pass
        else:
            assert False
    else:
        result = translate_filter(raw_filter)
        expr = 'artist in ("Taylor Swift", "Katy Perry")'
        assert expr == result
