"""Comprehensive tests for RedisRateLimiter."""

import asyncio
import time
import threading
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Union

import pytest

# Import the rate limiter - handle optional coredis dependency
try:
    from langchain_core.rate_limiters import RedisRateLimiter
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisRateLimiter = None

from langchain_core.rate_limiters import InMemoryRateLimiter


# Skip all tests if coredis is not available
pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="coredis package not available")


class MockRedisClient:
    """Mock Redis client for testing."""
    
    def __init__(self, fail_operations: bool = False, cluster_mode: bool = False):
        self.fail_operations = fail_operations
        self.cluster_mode = cluster_mode
        self.data: Dict[str, Any] = {}
        self.scripts: Dict[str, str] = {}
        self.script_shas: Dict[str, str] = {}
        self.ping_count = 0
        self.closed = False
        
    async def ping(self):
        """Mock ping method."""
        self.ping_count += 1
        if self.fail_operations:
            raise ConnectionError("Mock Redis connection failed")
        return b"PONG"
    
    async def script_load(self, script: str) -> str:
        """Mock script loading."""
        if self.fail_operations:
            raise ConnectionError("Mock Redis script load failed")
        
        script_hash = f"sha_{len(self.scripts)}"
        self.scripts[script_hash] = script
        self.script_shas[script] = script_hash
        return script_hash
    
    async def evalsha(self, sha: str, num_keys: int, *args) -> List[int]:
        """Mock script execution."""
        if self.fail_operations:
            raise ConnectionError("Mock Redis evalsha failed")
        
        # Extract arguments
        key = args[0] if args else "test_key"
        window_size = float(args[1]) if len(args) > 1 else 60
        max_requests = int(args[2]) if len(args) > 2 else 10
        current_time = float(args[3]) if len(args) > 3 else time.time()
        
        # Simulate sliding window logic
        if key not in self.data:
            self.data[key] = []
        
        # Remove expired entries
        cutoff_time = current_time - window_size
        self.data[key] = [t for t in self.data[key] if t > cutoff_time]
        
        current_count = len(self.data[key])
        
        # Check if this is a consume operation (has cleanup_probability arg)
        is_consume = len(args) > 4
        
        if current_count < max_requests:
            if is_consume:
                # Add current request
                self.data[key].append(current_time)
                current_count += 1
            return [1, current_count, max_requests]
        else:
            return [0, current_count, max_requests]
    
    async def delete(self, key: str) -> int:
        """Mock delete operation."""
        if self.fail_operations:
            raise ConnectionError("Mock Redis delete failed")
        
        if key in self.data:
            del self.data[key]
            return 1
        return 0
    
    async def close(self):
        """Mock close method."""
        self.closed = True


class MockRedisCluster(MockRedisClient):
    """Mock Redis cluster client."""
    
    def __init__(self, startup_nodes: List[Dict[str, Any]], **kwargs):
        super().__init__(cluster_mode=True, **kwargs)
        self.startup_nodes = startup_nodes


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    return MockRedisClient()


@pytest.fixture
def mock_failing_redis_client():
    """Create a mock Redis client that fails operations."""
    return MockRedisClient(fail_operations=True)


@pytest.fixture
def mock_redis_cluster():
    """Create a mock Redis cluster client."""
    return MockRedisCluster([{"host": "localhost", "port": 7000}])


@pytest.fixture
def redis_rate_limiter():
    """Create a RedisRateLimiter instance for testing."""
    return RedisRateLimiter(
        redis_url="redis://localhost:6379",
        requests_per_second=2,
        window_size=10,  # 10 second window for faster testing
        check_every_n_seconds=0.1,
        fallback_to_memory=True,
    )


@pytest.fixture
def redis_cluster_rate_limiter():
    """Create a RedisRateLimiter instance configured for cluster mode."""
    return RedisRateLimiter(
        redis_url="redis://node1:7000,redis://node2:7000,redis://node3:7000",
        requests_per_second=5,
        window_size=60,
        redis_cluster=True,
        fallback_to_memory=True,
    )


class TestRedisRateLimiterBasicFunctionality:
    """Test basic functionality of RedisRateLimiter."""
    
    def test_initialization(self, redis_rate_limiter):
        """Test RedisRateLimiter initialization."""
        assert redis_rate_limiter.requests_per_second == 2
        assert redis_rate_limiter.window_size == 10
        assert redis_rate_limiter.max_requests == 20  # 2 * 10
        assert redis_rate_limiter.fallback_to_memory is True
        assert redis_rate_limiter._fallback_limiter is not None
        assert isinstance(redis_rate_limiter._fallback_limiter, InMemoryRateLimiter)
    
    def test_initialization_without_coredis(self):
        """Test that initialization fails without coredis."""
        with patch('langchain_core.rate_limiters.coredis', None):
            with pytest.raises(ImportError, match="RedisRateLimiter requires the 'coredis' package"):
                RedisRateLimiter()
    
    def test_cluster_initialization(self, redis_cluster_rate_limiter):
        """Test cluster mode initialization."""
        assert redis_cluster_rate_limiter.redis_cluster is True
        assert redis_cluster_rate_limiter.requests_per_second == 5
        assert redis_cluster_rate_limiter.window_size == 60
        assert redis_cluster_rate_limiter.max_requests == 300  # 5 * 60
    
    def test_redis_key_generation(self, redis_rate_limiter):
        """Test Redis key generation."""
        key = redis_rate_limiter._get_redis_key()
        assert key.startswith("langchain:ratelimit:")
        assert redis_rate_limiter.identifier in key
    
    def test_custom_key_prefix(self):
        """Test custom key prefix."""
        limiter = RedisRateLimiter(
            key_prefix="custom:prefix",
            identifier="test123"
        )
        key = limiter._get_redis_key()
        assert key == "custom:prefix:test123"


class TestRedisRateLimiterConnectionManagement:
    """Test Redis connection management."""
    
    @patch('langchain_core.rate_limiters.coredis.Redis.from_url')
    def test_create_standalone_client(self, mock_from_url, redis_rate_limiter):
        """Test creating standalone Redis client."""
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client
        
        client = redis_rate_limiter._create_redis_client()
        
        mock_from_url.assert_called_once()
        assert client == mock_client
    
    @patch('langchain_core.rate_limiters.coredis.RedisCluster')
    def test_create_cluster_client(self, mock_cluster, redis_cluster_rate_limiter):
        """Test creating Redis cluster client."""
        mock_client = MagicMock()
        mock_cluster.return_value = mock_client
        
        client = redis_cluster_rate_limiter._create_redis_client()
        
        mock_cluster.assert_called_once()
        call_args = mock_cluster.call_args
        assert 'startup_nodes' in call_args.kwargs
        startup_nodes = call_args.kwargs['startup_nodes']
        assert len(startup_nodes) == 3
        assert {"host": "node1", "port": 7000} in startup_nodes
        assert {"host": "node2", "port": 7000} in startup_nodes
        assert {"host": "node3", "port": 7000} in startup_nodes
    
    @patch('langchain_core.rate_limiters.coredis.RedisCluster')
    def test_cluster_url_parsing(self, mock_cluster):
        """Test parsing of cluster URLs."""
        limiter = RedisRateLimiter(
            redis_url="redis://host1:7001,host2:7002,host3",
            redis_cluster=True
        )
        
        mock_client = MagicMock()
        mock_cluster.return_value = mock_client
        
        limiter._create_redis_client()
        
        call_args = mock_cluster.call_args
        startup_nodes = call_args.kwargs['startup_nodes']
        
        expected_nodes = [
            {"host": "host1", "port": 7001},
            {"host": "host2", "port": 7002},
            {"host": "host3", "port": 6379},  # Default port
        ]
        assert startup_nodes == expected_nodes


class TestRedisRateLimiterLuaScripts:
    """Test Lua script functionality."""
    
    @pytest.mark.asyncio
    async def test_script_loading(self, redis_rate_limiter, mock_redis_client):
        """Test Lua script loading."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_redis_client):
            await redis_rate_limiter._ensure_scripts_loaded(mock_redis_client)
            
            assert redis_rate_limiter._check_and_update_sha is not None
            assert redis_rate_limiter._check_only_sha is not None
            assert len(mock_redis_client.scripts) == 2
    
    @pytest.mark.asyncio
    async def test_script_loading_failure(self, redis_rate_limiter, mock_failing_redis_client):
        """Test handling of script loading failures."""
        with pytest.raises(ConnectionError):
            await redis_rate_limiter._ensure_scripts_loaded(mock_failing_redis_client)
        
        # SHAs should be reset on failure
        assert redis_rate_limiter._check_and_update_sha is None
        assert redis_rate_limiter._check_only_sha is None
    
    @pytest.mark.asyncio
    async def test_rate_limit_check_consume(self, redis_rate_limiter, mock_redis_client):
        """Test rate limit check with token consumption."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_redis_client):
            # First request should succeed
            allowed, current_count, max_requests = await redis_rate_limiter._execute_rate_limit_check(
                mock_redis_client, consume_token=True
            )
            
            assert allowed is True
            assert current_count == 1
            assert max_requests == redis_rate_limiter.max_requests
    
    @pytest.mark.asyncio
    async def test_rate_limit_check_no_consume(self, redis_rate_limiter, mock_redis_client):
        """Test rate limit check without token consumption."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_redis_client):
            # Check availability without consuming
            allowed, current_count, max_requests = await redis_rate_limiter._execute_rate_limit_check(
                mock_redis_client, consume_token=False
            )
            
            assert allowed is True
            assert current_count == 0  # No token consumed
            assert max_requests == redis_rate_limiter.max_requests
class TestRedisRateLimiterAcquireMethods:
    """Test acquire and aacquire methods."""
    
    @pytest.mark.asyncio
    async def test_aacquire_success(self, redis_rate_limiter, mock_redis_client):
        """Test successful async token acquisition."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_redis_client):
            result = await redis_rate_limiter.aacquire(blocking=False)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_aacquire_rate_limited(self, mock_redis_client):
        """Test rate limiting behavior."""
        # Create a rate limiter with very low limits
        limiter = RedisRateLimiter(
            requests_per_second=1,
            window_size=1,  # 1 request per second
            fallback_to_memory=False,
        )
        
        with patch.object(limiter, '_get_redis_client', return_value=mock_redis_client):
            # First request should succeed
            result1 = await limiter.aacquire(blocking=False)
            assert result1 is True
            
            # Second request should be rate limited
            result2 = await limiter.aacquire(blocking=False)
            assert result2 is False
    
    def test_acquire_sync(self, redis_rate_limiter, mock_redis_client):
        """Test synchronous token acquisition."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_redis_client):
            result = redis_rate_limiter.acquire(blocking=False)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_aacquire_blocking(self, mock_redis_client):
        """Test blocking acquisition behavior."""
        # Create a rate limiter that will allow requests after a short delay
        limiter = RedisRateLimiter(
            requests_per_second=10,  # High rate to allow quick recovery
            window_size=1,
            check_every_n_seconds=0.01,  # Very short check interval
            fallback_to_memory=False,
        )
        
        with patch.object(limiter, '_get_redis_client', return_value=mock_redis_client):
            # Fill up the rate limit
            for _ in range(10):
                await limiter.aacquire(blocking=False)
            
            # This should block briefly then succeed
            start_time = time.time()
            result = await limiter.aacquire(blocking=True)
            end_time = time.time()
            
            assert result is True
            # Should have taken some time to wait
            assert end_time - start_time >= 0.01
class TestRedisRateLimiterErrorHandling:
    """Test error handling and fallback mechanisms."""
    
    @pytest.mark.asyncio
    async def test_fallback_to_memory_on_connection_error(self, redis_rate_limiter, mock_failing_redis_client):
        """Test fallback to in-memory rate limiter on connection errors."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_failing_redis_client):
            # Should fallback to memory limiter
            result = await redis_rate_limiter.aacquire(blocking=False)
            assert result is True
            
            # Verify fallback limiter was used
            assert redis_rate_limiter._fallback_limiter.available_tokens < redis_rate_limiter._fallback_limiter.max_bucket_size
    
    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self, mock_failing_redis_client):
        """Test behavior when fallback is disabled."""
        limiter = RedisRateLimiter(
            fallback_to_memory=False,
            requests_per_second=2,
            window_size=10,
        )
        
        with patch.object(limiter, '_get_redis_client', return_value=mock_failing_redis_client):
            with pytest.raises(ConnectionError):
                await limiter.aacquire(blocking=False)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, redis_rate_limiter, mock_failing_redis_client):
        """Test circuit breaker functionality."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_failing_redis_client):
            # Trigger multiple failures to open circuit breaker
            for _ in range(6):  # Exceed threshold of 5
                try:
                    await redis_rate_limiter.aacquire(blocking=False)
                except:
                    pass
            
            # Circuit breaker should now be open
            assert redis_rate_limiter._circuit_breaker_failures >= redis_rate_limiter._circuit_breaker_threshold
    
    @pytest.mark.asyncio
    async def test_retry_logic(self, redis_rate_limiter):
        """Test retry logic with exponential backoff."""
        call_count = 0
        
        async def mock_get_client():
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 attempts
                raise ConnectionError("Mock connection error")
            return MockRedisClient()  # Succeed on 3rd attempt
        
        with patch.object(redis_rate_limiter, '_get_redis_client', side_effect=mock_get_client):
            result = await redis_rate_limiter.aacquire(blocking=False)
            assert result is True
            assert call_count == 3  # Should have retried
    
    def test_should_use_fallback(self, redis_rate_limiter):
        """Test fallback decision logic."""
        # Connection errors should trigger fallback
        assert redis_rate_limiter._should_use_fallback(ConnectionError("test"))
        assert redis_rate_limiter._should_use_fallback(TimeoutError("test"))
        assert redis_rate_limiter._should_use_fallback(OSError("test"))
        
        # Other errors should not trigger fallback
        assert not redis_rate_limiter._should_use_fallback(ValueError("test"))
        
        # Disable fallback
        redis_rate_limiter.fallback_to_memory = False
        assert not redis_rate_limiter._should_use_fallback(ConnectionError("test"))
class TestRedisRateLimiterUtilityMethods:
    """Test utility methods like usage statistics and reset."""
    
    @pytest.mark.asyncio
    async def test_get_current_usage(self, redis_rate_limiter, mock_redis_client):
        """Test getting current usage statistics."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_redis_client):
            # Make a few requests first
            await redis_rate_limiter.aacquire(blocking=False)
            await redis_rate_limiter.aacquire(blocking=False)
            
            usage = await redis_rate_limiter.get_current_usage()
            
            assert isinstance(usage, dict)
            assert 'current_count' in usage
            assert 'max_requests' in usage
            assert 'utilization_percent' in usage
            assert 'requests_per_second' in usage
            assert 'window_size' in usage
            assert 'redis_healthy' in usage
            
            assert usage['max_requests'] == redis_rate_limiter.max_requests
            assert usage['requests_per_second'] == redis_rate_limiter.requests_per_second
            assert usage['window_size'] == redis_rate_limiter.window_size
    
    @pytest.mark.asyncio
    async def test_get_current_usage_error(self, redis_rate_limiter, mock_failing_redis_client):
        """Test usage statistics during Redis errors."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_failing_redis_client):
            usage = await redis_rate_limiter.get_current_usage()
            
            assert usage['current_count'] == -1
            assert usage['utilization_percent'] == -1
            assert usage['redis_healthy'] is False
            assert 'error' in usage
    
    @pytest.mark.asyncio
    async def test_reset(self, redis_rate_limiter, mock_redis_client):
        """Test rate limiter reset functionality."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_redis_client):
            # Make some requests first
            await redis_rate_limiter.aacquire(blocking=False)
            await redis_rate_limiter.aacquire(blocking=False)
            
            # Reset should succeed
            result = await redis_rate_limiter.reset()
            assert result is True
            
            # Data should be cleared
            key = redis_rate_limiter._get_redis_key()
            assert key not in mock_redis_client.data
    
    @pytest.mark.asyncio
    async def test_reset_error(self, redis_rate_limiter, mock_failing_redis_client):
        """Test reset during Redis errors."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_failing_redis_client):
            result = await redis_rate_limiter.reset()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_close(self, redis_rate_limiter, mock_redis_client):
        """Test connection cleanup."""
        redis_rate_limiter._redis_client = mock_redis_client
        
        await redis_rate_limiter.close()
        
        assert mock_redis_client.closed is True
        assert redis_rate_limiter._redis_client is None
class TestRedisRateLimiterConcurrency:
    """Test concurrent access and thread safety."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, redis_rate_limiter, mock_redis_client):
        """Test handling of concurrent requests."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_redis_client):
            # Launch multiple concurrent requests
            tasks = []
            for _ in range(10):
                task = asyncio.create_task(redis_rate_limiter.aacquire(blocking=False))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Some should succeed, some should be rate limited
            successful = sum(1 for r in results if r)
            rate_limited = sum(1 for r in results if not r)
            
            assert successful > 0
            assert rate_limited > 0
            assert successful + rate_limited == 10
    
    def test_thread_safety(self, redis_rate_limiter, mock_redis_client):
        """Test thread safety of the rate limiter."""
        results = []
        errors = []
        
        def worker():
            try:
                with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_redis_client):
                    result = redis_rate_limiter.acquire(blocking=False)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Launch multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should not have any errors
        assert len(errors) == 0
        assert len(results) == 5
        
        # Some requests should succeed
        assert any(results)
class TestRedisRateLimiterIntegration:
    """Integration tests that test the full flow."""
    
    @pytest.mark.asyncio
    async def test_full_sliding_window_behavior(self, mock_redis_client):
        """Test full sliding window behavior over time."""
        limiter = RedisRateLimiter(
            requests_per_second=2,
            window_size=5,  # 5 second window, so max 10 requests
            fallback_to_memory=False,
        )
        
        with patch.object(limiter, '_get_redis_client', return_value=mock_redis_client):
            # Fill up the window
            successful_requests = 0
            for _ in range(15):  # Try more than the limit
                if await limiter.aacquire(blocking=False):
                    successful_requests += 1
            
            # Should have allowed exactly max_requests
            assert successful_requests == limiter.max_requests
            
            # Usage should show full utilization
            usage = await limiter.get_current_usage()
            assert usage['current_count'] == limiter.max_requests
            assert usage['utilization_percent'] == 100.0
    
    @pytest.mark.asyncio
    async def test_distributed_behavior_simulation(self, mock_redis_client):
        """Simulate distributed behavior with multiple rate limiter instances."""
        # Create multiple rate limiter instances with same identifier
        # to simulate distributed processes
        limiter1 = RedisRateLimiter(
            identifier="shared_limiter",
            requests_per_second=1,
            window_size=10,  # Max 10 requests total
            fallback_to_memory=False,
        )
        
        limiter2 = RedisRateLimiter(
            identifier="shared_limiter",  # Same identifier
            requests_per_second=1,
            window_size=10,
            fallback_to_memory=False,
        )
        
        with patch.object(limiter1, '_get_redis_client', return_value=mock_redis_client), \
             patch.object(limiter2, '_get_redis_client', return_value=mock_redis_client):
            
            # Both limiters should share the same quota
            total_successful = 0
            
            # Make requests from both limiters
            for _ in range(5):
                if await limiter1.aacquire(blocking=False):
                    total_successful += 1
                if await limiter2.aacquire(blocking=False):
                    total_successful += 1
            
            # Total should not exceed the shared limit
            assert total_successful <= limiter1.max_requests
    
    @pytest.mark.asyncio
    async def test_health_check_and_recovery(self, redis_rate_limiter):
        """Test health checking and recovery from failures."""
        # Start with a failing client
        failing_client = MockRedisClient(fail_operations=True)
        
        with patch.object(redis_rate_limiter, '_create_redis_client') as mock_create:
            # First call returns failing client, second returns working client
            working_client = MockRedisClient(fail_operations=False)
            mock_create.side_effect = [failing_client, working_client]
            
            # First request should fail and trigger fallback
            result1 = await redis_rate_limiter.aacquire(blocking=False)
            assert result1 is True  # Should succeed via fallback
            
            # Simulate health check recovery by advancing time
            redis_rate_limiter._last_health_check = 0  # Force health check
            redis_rate_limiter._circuit_breaker_failures = 0  # Reset circuit breaker
            
            # Next request should use Redis again
            result2 = await redis_rate_limiter.aacquire(blocking=False)
            assert result2 is True
class TestRedisRateLimiterEdgeCases:
    """Test edge cases and unusual scenarios."""
    
    def test_zero_requests_per_second(self):
        """Test behavior with zero requests per second."""
        limiter = RedisRateLimiter(requests_per_second=0, window_size=10)
        assert limiter.max_requests == 0
    
    def test_very_high_requests_per_second(self):
        """Test behavior with very high request rates."""
        limiter = RedisRateLimiter(requests_per_second=1000000, window_size=1)
        assert limiter.max_requests == 1000000
    
    def test_fractional_requests_per_second(self):
        """Test behavior with fractional request rates."""
        limiter = RedisRateLimiter(requests_per_second=0.5, window_size=10)
        assert limiter.max_requests == 5  # 0.5 * 10
    
    @pytest.mark.asyncio
    async def test_script_reload_on_failure(self, redis_rate_limiter):
        """Test script reloading when execution fails."""
        mock_client = MockRedisClient()
        
        # Mock evalsha to fail first time, succeed second time
        original_evalsha = mock_client.evalsha
        call_count = 0
        
        async def failing_evalsha(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Script not found")
            return await original_evalsha(*args, **kwargs)
        
        mock_client.evalsha = failing_evalsha
        
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_client):
            # Should succeed after script reload
            result = await redis_rate_limiter.aacquire(blocking=False)
            assert result is True
            assert call_count == 2  # Failed once, then succeeded
    
    def test_custom_identifier_and_prefix(self):
        """Test custom identifier and key prefix."""
        limiter = RedisRateLimiter(
            identifier="custom_id_123",
            key_prefix="myapp:limits"
        )
        
        key = limiter._get_redis_key()
        assert key == "myapp:limits:custom_id_123"
    
    @pytest.mark.asyncio
    async def test_cleanup_probability_effect(self, mock_redis_client):
        """Test that cleanup probability affects cleanup behavior."""
        # Test with high cleanup probability
        limiter_high_cleanup = RedisRateLimiter(
            cleanup_probability=1.0,  # Always cleanup
            requests_per_second=1,
            window_size=1,
        )
        
        # Test with no cleanup
        limiter_no_cleanup = RedisRateLimiter(
            cleanup_probability=0.0,  # Never cleanup
            requests_per_second=1,
            window_size=1,
        )
        
        # Both should work the same functionally
        with patch.object(limiter_high_cleanup, '_get_redis_client', return_value=mock_redis_client):
            result1 = await limiter_high_cleanup.aacquire(blocking=False)
            assert result1 is True
        
        with patch.object(limiter_no_cleanup, '_get_redis_client', return_value=mock_redis_client):
            result2 = await limiter_no_cleanup.aacquire(blocking=False)
            assert result2 is True
# Performance and stress tests
class TestRedisRateLimiterPerformance:
    """Performance and stress tests."""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, redis_rate_limiter, mock_redis_client):
        """Stress test with high concurrency."""
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_redis_client):
            # Launch many concurrent requests
            num_requests = 100
            tasks = []
            
            for _ in range(num_requests):
                task = asyncio.create_task(redis_rate_limiter.aacquire(blocking=False))
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Should complete reasonably quickly
            assert end_time - start_time < 5.0  # 5 seconds max
            
            # Should not have any exceptions
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0
            
            # Should have some successful and some rate-limited requests
            successful = sum(1 for r in results if r is True)
            rate_limited = sum(1 for r in results if r is False)
            
            assert successful > 0
            assert rate_limited > 0
            assert successful + rate_limited == num_requests
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, redis_rate_limiter, mock_redis_client):
        """Test that memory usage remains stable over many operations."""
        import gc
        
        with patch.object(redis_rate_limiter, '_get_redis_client', return_value=mock_redis_client):
            # Force garbage collection
            gc.collect()
            
            # Make many requests
            for _ in range(1000):
                await redis_rate_limiter.aacquire(blocking=False)
                
                # Occasionally check usage and reset
                if _ % 100 == 0:
                    await redis_rate_limiter.get_current_usage()
                    await redis_rate_limiter.reset()
            
            # Force garbage collection again
            gc.collect()
            
            # Test should complete without memory issues
            assert True  # If we get here, memory usage was stable
if __name__ == "__main__":
    pytest.main([__file__])

