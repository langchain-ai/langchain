"""Unit tests for the RustChain LangChain tool."""

from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import patch


def _fake_health(*args, **kwargs):
    return {"ok": True, "version": "2.2.1-rip200"}


def _fake_epoch(*args, **kwargs):
    return {"epoch": 249, "slot": 7, "blocks_per_epoch": 8}


def _fake_balance(*args, **kwargs):
    return {"miner_id": "demo", "amount_i64": 0, "amount_rtc": 0.0}


def _fake_bounties(limit=10):
    return [
        {"number": 16275, "title": "Fix docs", "url": "https://github.com/Scottcjn/rustchain-bounties/issues/16275", "labels": ["bounty"]},
        {"number": 16271, "title": "Add test", "url": "https://github.com/Scottcjn/rustchain-bounties/issues/16271", "labels": ["bounty"]},
    ][:limit]


class RustChainToolTests(unittest.TestCase):
    def setUp(self) -> None:
        from langchain_rustchain.tools import RustChainTool

        self.tool = RustChainTool()

    def test_check_balance(self):
        with patch("langchain_rustchain.tools._get", side_effect=_fake_balance):
            self.assertEqual(self.tool.check_balance("demo"), 0.0)

    def test_list_bounties(self):
        with patch("langchain_rustchain.tools._get_bounties", side_effect=_fake_bounties):
            b = self.tool.list_bounties(2)
            self.assertEqual(len(b), 2)
            self.assertIn("number", b[0])

    def test_get_node_health(self):
        with patch("langchain_rustchain.tools._get", side_effect=_fake_health):
            h = self.tool.get_node_health()
            self.assertTrue(h["ok"])

    def test_get_current_epoch(self):
        with patch("langchain_rustchain.tools._get", side_effect=_fake_epoch):
            e = self.tool.get_current_epoch()
            self.assertEqual(e["epoch"], 249)

    def test_run_through_base_tool(self):
        with patch("langchain_rustchain.tools._get", side_effect=_fake_health):
            out = json.loads(self.tool._run('{"method": "get_node_health"}'))
            self.assertTrue(out["ok"])

    def test_unknown_method(self):
        out = self.tool._run({"method": "nope"})
        self.assertIn("unknown method", out)

    def test_invalid_json(self):
        out = self.tool._run("not-json")
        self.assertIn("error", json.loads(out))

    def test_nombre_y_descripcion(self):
        self.assertEqual(self.tool.name, "rustchain")
        self.assertIn("balance", self.tool.description)


if __name__ == "__main__":
    unittest.main()