from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware


def test_glob_pattern_matches():
    mw = HumanInTheLoopMiddleware(interrupt_on={"db_write_*": True})
    cfg = mw._resolve_config_for("db_write_custom")
    assert cfg is not None
    assert "approve" in cfg["allowed_decisions"]


def test_regex_pattern_matches():
    mw = HumanInTheLoopMiddleware(interrupt_on={"re:^payment_\\d+$": True})
    cfg = mw._resolve_config_for("payment_123")
    assert cfg is not None
    assert "approve" in cfg["allowed_decisions"]


def test_exact_precedence_over_pattern():
    mw = HumanInTheLoopMiddleware(
        interrupt_on={
            "payment_processor": {"allowed_decisions": ["approve"]},
            "payment_*": {"allowed_decisions": ["reject"]},
        }
    )
    cfg = mw._resolve_config_for("payment_processor")
    assert cfg is not None
    assert cfg["allowed_decisions"] == ["approve"]


def test_pattern_order_first_match_wins():
    mw = HumanInTheLoopMiddleware(
        interrupt_on={
            "pay*": {"allowed_decisions": ["first"]},
            "payment_*": {"allowed_decisions": ["second"]},
        }
    )
    cfg = mw._resolve_config_for("payment_99")
    assert cfg is not None
    assert cfg["allowed_decisions"] == ["first"]
