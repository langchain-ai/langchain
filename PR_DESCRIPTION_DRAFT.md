## 📌 Summary
Fixes #39713

Preserves `response_metadata` fields (`model_name`, `stop_reason`, `stop_sequence`, etc.) when `stream_usage=False` in `ChatAnthropic`, while properly omitting only `usage_metadata`.

---

## 🔍 Root Cause Analysis
Previously in `ChatAnthropic._make_message_chunk_from_anthropic_event`, `stream_usage` was part of the event-dispatch conditions (`if event.type == "message_start" and stream_usage:` and `elif event.type == "message_delta" and stream_usage:`). When callers set `stream_usage=False`, neither branch produced an `AIMessageChunk`, which caused callers to lose `model_name` and `stop_reason` during streaming completions.

---

## 🛠️ Solution
- Decoupled `stream_usage` from event handling in `message_start` and `message_delta`.
- Gated only the creation of `usage_metadata` (`_create_usage_metadata(event.usage) if stream_usage else None`).
- Added unit test in `libs/partners/anthropic/tests/unit_tests/test_stream_usage_metadata.py`.
