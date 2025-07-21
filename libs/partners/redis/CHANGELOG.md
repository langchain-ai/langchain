# Changelog

## Version 0.1.0 (2025-01-XX)

### Added
- Initial release of langchain-redis integration package
- `RedisChatMessageHistory` class for storing chat message history in Redis
- Support for session-based storage with optional key prefixes
- TTL (Time-To-Live) support for Redis keys
- Comprehensive unit and integration tests
- Full type hints and error handling

### Fixed
- **Issue #30535**: Fixed bug where `key_prefix` parameter prevented message retrieval
  - Messages are now correctly stored and retrieved when using key_prefix
  - Redis key construction is consistent between storage and retrieval operations
  - Proper key format: `{key_prefix}{session_id}`

### Features
- **Session Management**: Store and retrieve messages by session ID
- **Key Prefixes**: Organize sessions with optional key prefixes (e.g., "chat_app:")
- **TTL Support**: Set expiration time for Redis keys
- **Error Handling**: Graceful error handling with logging
- **Type Safety**: Full type hints for better development experience
- **Async Support**: Inherits async methods from BaseChatMessageHistory

### Compatibility
- Python 3.9+
- Redis 4.0+
- LangChain Core 0.3.60+