# RedisRateLimiter Design Specification

## Overview

This document outlines the design for a high-performance, distributed RedisRateLimiter that extends the BaseRateLimiter interface. The implementation uses the coredis package to support Redis clusters and provides distributed rate limiting across multiple processes, pods, and containers.

## Design Goals

1. **Distributed**: Share rate limiting state across multiple processes/pods/containers
2. **High Performance**: Minimize Redis round trips and use atomic operations
3. **Redis Cluster Support**: Leverage coredis for cluster compatibility
4. **Interface Compatibility**: Maintain BaseRateLimiter interface
5. **Fault Tolerance**: Graceful degradation and error handling
6. **Scalable**: Support high-frequency rate limiting scenarios
7. **Configurable**: Flexible algorithm and parameter configuration

## Algorithm Selection: Sliding Window Counter

### Why Sliding Window Over Token Bucket?

1. **Better Distributed Behavior**: Token bucket requires frequent state updates for token refill, creating more Redis operations
2. **Atomic Operations**: Sliding window can be implemented with single Redis operations using Lua scripts
3. **Memory Efficiency**: Uses Redis sorted sets with automatic cleanup of old entries
4. **Precise Rate Limiting**: More accurate rate limiting without the "burst" behavior of token buckets
5. **Simpler State Management**: No need to track last refill time across distributed instances

### Sliding Window Implementation

- Use Redis sorted sets (ZSET) to store request timestamps
- Score = timestamp, Member = unique request ID
- Window slides by removing entries older than the time window
- Rate limit = count of entries in current window

## Redis Data Structures

### Primary Key Structure
```
rate_limiter:{limiter_id}:{window_start_time}
```

### Data Storage
- **Type**: Redis Sorted Set (ZSET)
- **Score**: Request timestamp (microseconds since epoch)
- **Member**: Unique request identifier (UUID + timestamp)
- **TTL**: Set to window_size + buffer to auto-cleanup old keys

### Key Benefits
- Automatic expiration prevents memory leaks
- Sorted set operations are O(log N)
- Atomic operations via Lua scripts
- Cluster-friendly key distribution

## Lua Scripts for Atomic Operations

### Script 1: Sliding Window Rate Check and Increment
```lua
-- KEYS[1]: rate limiter key
-- ARGV[1]: current timestamp (microseconds)
-- ARGV[2]: window size (microseconds)  
-- ARGV[3]: max requests per window
-- ARGV[4]: request identifier
-- ARGV[5]: key TTL (seconds)

local key = KEYS[1]
local now = tonumber(ARGV[1])
local window_size = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local request_id = ARGV[4]
local ttl = tonumber(ARGV[5])

-- Remove expired entries
local window_start = now - window_size
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

-- Count current requests in window
local current_count = redis.call('ZCARD', key)

-- Check if we can proceed
if current_count < max_requests then
    -- Add current request
    redis.call('ZADD', key, now, request_id)
    -- Set TTL if key is new
    redis.call('EXPIRE', key, ttl)
    return {1, current_count + 1}  -- {allowed, new_count}
else
    return {0, current_count}  -- {denied, current_count}
end
```

### Script 2: Non-blocking Rate Check (Read-only)
```lua
-- KEYS[1]: rate limiter key
-- ARGV[1]: current timestamp (microseconds)
-- ARGV[2]: window size (microseconds)
-- ARGV[3]: max requests per window

local key = KEYS[1]
local now = tonumber(ARGV[1])
local window_size = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])

-- Remove expired entries (cleanup)
local window_start = now - window_size
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

-- Count current requests
local current_count = redis.call('ZCARD', key)

-- Return availability without consuming
if current_count < max_requests then
    return {1, current_count}  -- {available, current_count}
else
    return {0, current_count}  -- {not_available, current_count}
end
```

## Connection Management

### Connection Pooling Strategy
- Use coredis connection pools for efficient connection reuse
- Separate pools for different Redis instances/clusters
- Configurable pool size based on expected concurrency
- Connection health monitoring and automatic reconnection

### Redis Cluster Support
- Leverage coredis cluster client for automatic node discovery
- Hash slot-aware key distribution
- Automatic failover and node addition/removal handling
- Consistent hashing for rate limiter key distribution

## Class Architecture

### RedisRateLimiter Class
```python
class RedisRateLimiter(BaseRateLimiter):
    def __init__(
        self,
        *,
        requests_per_second: float = 1.0,
        window_size_seconds: float = 1.0,
        redis_url: Optional[str] = None,
        redis_client: Optional[Redis] = None,
        limiter_id: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        fallback_to_memory: bool = True,
        connection_pool_size: int = 10,
        cluster_mode: bool = False,
    )
```

### Key Parameters
- **requests_per_second**: Rate limit (converted to requests per window)
- **window_size_seconds**: Sliding window duration
- **redis_url/redis_client**: Redis connection configuration
- **limiter_id**: Unique identifier for this rate limiter instance
- **max_retries**: Redis operation retry count
- **retry_delay**: Delay between retries
- **fallback_to_memory**: Fall back to InMemoryRateLimiter on Redis failure
- **connection_pool_size**: Redis connection pool size
- **cluster_mode**: Enable Redis cluster support

## Error Handling Strategy

### Redis Connection Failures
1. **Retry Logic**: Exponential backoff with configurable max retries
2. **Circuit Breaker**: Temporarily disable Redis after consecutive failures
3. **Fallback Mode**: Optional fallback to InMemoryRateLimiter
4. **Health Monitoring**: Periodic Redis health checks

### Partial Failures
- Handle Redis timeouts gracefully
- Distinguish between temporary and permanent failures
- Logging and metrics for failure analysis

### Cluster Failures
- Handle node failures and cluster reconfiguration
- Automatic retry with different cluster nodes
- Graceful degradation during cluster maintenance

## Performance Optimizations

### Lua Script Optimization
- Pre-compiled and cached Lua scripts
- Minimal Redis operations per request
- Efficient data structure operations

### Connection Optimization
- Connection pooling and reuse
- Pipeline operations where possible
- Async/await support for non-blocking operations

### Memory Optimization
- Automatic cleanup of expired entries
- Configurable TTL for rate limiter keys
- Efficient key naming to minimize memory usage

### Network Optimization
- Single Redis round trip per rate limit check
- Batch operations for multiple rate limiters
- Compression for large payloads (if needed)

## Interface Compatibility

### BaseRateLimiter Methods
```python
def acquire(self, *, blocking: bool = True) -> bool:
    """Sync version with Redis operations"""

async def aacquire(self, *, blocking: bool = True) -> bool:
    """Async version with Redis operations"""
```

### Additional Methods
```python
def get_current_usage(self) -> int:
    """Get current request count in window"""

def reset(self) -> None:
    """Reset rate limiter state"""

async def aget_current_usage(self) -> int:
    """Async version of get_current_usage"""

async def areset(self) -> None:
    """Async version of reset"""
```

## Configuration Examples

### Basic Usage
```python
rate_limiter = RedisRateLimiter(
    requests_per_second=10.0,
    redis_url="redis://localhost:6379"
)
```

### Cluster Configuration
```python
rate_limiter = RedisRateLimiter(
    requests_per_second=100.0,
    redis_url="redis://cluster-node1:6379,cluster-node2:6379",
    cluster_mode=True,
    connection_pool_size=20
)
```

This design provides a robust, high-performance, distributed rate limiting solution that maintains compatibility with the existing BaseRateLimiter interface while leveraging Redis for distributed state management and coredis for cluster support.

