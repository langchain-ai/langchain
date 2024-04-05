from collections import defaultdict

from langgraph.checkpoint import Checkpoint
from langgraph.checkpoint.base import CheckpointTuple

from langchain_postgres.checkpoint import PickleCheckpointSerializer, PostgresCheckpoint
from tests.utils import asyncpg_client, syncpg_client


async def test_async_checkpoint() -> None:
    """Test the async chat history."""
    async with asyncpg_client() as async_connection:
        await PostgresCheckpoint.adrop_schema(async_connection)
        await PostgresCheckpoint.acreate_schema(async_connection)
        checkpoint_saver = PostgresCheckpoint(
            async_connection=async_connection, serializer=PickleCheckpointSerializer()
        )
        checkpoint_tuple = [
            c
            async for c in checkpoint_saver.alist(
                {
                    "configurable": {
                        "thread_id": "test_thread",
                    }
                }
            )
        ]
        assert len(checkpoint_tuple) == 0

        # Add a checkpoint
        sample_checkpoint: Checkpoint = {
            "v": 1,
            "ts": "2021-09-01T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": defaultdict(),
            "versions_seen": defaultdict(),
        }

        await checkpoint_saver.aput(
            {
                "configurable": {
                    "thread_id": "test_thread",
                }
            },
            sample_checkpoint,
        )

        checkpoints = [
            c
            async for c in checkpoint_saver.alist(
                {
                    "configurable": {
                        "thread_id": "test_thread",
                    }
                }
            )
        ]

        assert len(checkpoints) == 1
        assert checkpoints[0].checkpoint == sample_checkpoint

        # Add another checkpoint
        sample_checkpoint2: Checkpoint = {
            "v": 1,
            "ts": "2021-09-02T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": defaultdict(),
            "versions_seen": defaultdict(),
        }

        await checkpoint_saver.aput(
            {
                "configurable": {
                    "thread_id": "test_thread",
                }
            },
            sample_checkpoint2,
        )

        # Try aget
        checkpoints = [
            c
            async for c in checkpoint_saver.alist(
                {
                    "configurable": {
                        "thread_id": "test_thread",
                    }
                }
            )
        ]

        assert len(checkpoints) == 2
        # Should be sorted by timestamp desc
        assert checkpoints[0].checkpoint == sample_checkpoint2
        assert checkpoints[1].checkpoint == sample_checkpoint

        assert await checkpoint_saver.aget_tuple(
            {
                "configurable": {
                    "thread_id": "test_thread",
                }
            }
        ) == CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "test_thread",
                    "thread_ts": "2021-09-02T00:00:00+00:00",
                }
            },
            checkpoint={
                "v": 1,
                "ts": "2021-09-02T00:00:00+00:00",
                "channel_values": {},
                "channel_versions": {},  # type: ignore
                "versions_seen": {},  # type: ignore
            },
            parent_config=None,
        )

        # Check aget_tuple with thread_ts
        assert await checkpoint_saver.aget_tuple(
            {
                "configurable": {
                    "thread_id": "test_thread",
                    "thread_ts": "2021-09-01T00:00:00+00:00",
                }
            }
        ) == CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "test_thread",
                    "thread_ts": "2021-09-01T00:00:00+00:00",
                }
            },
            checkpoint={
                "v": 1,
                "ts": "2021-09-01T00:00:00+00:00",
                "channel_values": {},
                "channel_versions": {},  # type: ignore
                "versions_seen": {},  # type: ignore
            },
            parent_config=None,
        )


def test_sync_checkpoint() -> None:
    """Test the sync check point implementation."""
    with syncpg_client() as sync_connection:
        PostgresCheckpoint.drop_schema(sync_connection)
        PostgresCheckpoint.create_schema(sync_connection)
        checkpoint_saver = PostgresCheckpoint(
            sync_connection=sync_connection, serializer=PickleCheckpointSerializer()
        )
        checkpoint_tuple = [
            c
            for c in checkpoint_saver.list(
                {
                    "configurable": {
                        "thread_id": "test_thread",
                    }
                }
            )
        ]
        assert len(checkpoint_tuple) == 0

        # Add a checkpoint
        sample_checkpoint: Checkpoint = {
            "v": 1,
            "ts": "2021-09-01T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": defaultdict(),
            "versions_seen": defaultdict(),
        }

        checkpoint_saver.put(
            {
                "configurable": {
                    "thread_id": "test_thread",
                }
            },
            sample_checkpoint,
        )

        checkpoints = [
            c
            for c in checkpoint_saver.list(
                {
                    "configurable": {
                        "thread_id": "test_thread",
                    }
                }
            )
        ]

        assert len(checkpoints) == 1
        assert checkpoints[0].checkpoint == sample_checkpoint

        # Add another checkpoint
        sample_checkpoint_2: Checkpoint = {
            "v": 1,
            "ts": "2021-09-02T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": defaultdict(),
            "versions_seen": defaultdict(),
        }

        checkpoint_saver.put(
            {
                "configurable": {
                    "thread_id": "test_thread",
                }
            },
            sample_checkpoint_2,
        )

        # Try aget
        checkpoints = [
            c
            for c in checkpoint_saver.list(
                {
                    "configurable": {
                        "thread_id": "test_thread",
                    }
                }
            )
        ]

        assert len(checkpoints) == 2
        # Should be sorted by timestamp desc
        assert checkpoints[0].checkpoint == sample_checkpoint_2
        assert checkpoints[1].checkpoint == sample_checkpoint

        assert checkpoint_saver.get_tuple(
            {
                "configurable": {
                    "thread_id": "test_thread",
                }
            }
        ) == CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "test_thread",
                    "thread_ts": "2021-09-02T00:00:00+00:00",
                }
            },
            checkpoint={
                "v": 1,
                "ts": "2021-09-02T00:00:00+00:00",
                "channel_values": {},
                "channel_versions": defaultdict(),
                "versions_seen": defaultdict(),
            },
            parent_config=None,
        )


async def test_on_conflict_aput() -> None:
    async with asyncpg_client() as async_connection:
        await PostgresCheckpoint.adrop_schema(async_connection)
        await PostgresCheckpoint.acreate_schema(async_connection)
        checkpoint_saver = PostgresCheckpoint(
            async_connection=async_connection, serializer=PickleCheckpointSerializer()
        )

        # aput with twice on the same (thread_id, thread_ts) should not raise any error
        sample_checkpoint: Checkpoint = {
            "v": 1,
            "ts": "2021-09-01T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": defaultdict(),
            "versions_seen": defaultdict(),
        }
        new_checkpoint: Checkpoint = {
            "v": 2,
            "ts": "2021-09-01T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": defaultdict(),
            "versions_seen": defaultdict(),
        }
        await checkpoint_saver.aput(
            {
                "configurable": {
                    "thread_id": "test_thread",
                    "thread_ts": "2021-09-01T00:00:00+00:00",
                }
            },
            sample_checkpoint,
        )
        await checkpoint_saver.aput(
            {
                "configurable": {
                    "thread_id": "test_thread",
                    "thread_ts": "2021-09-01T00:00:00+00:00",
                }
            },
            new_checkpoint,
        )
        # Check aget_tuple with thread_ts
        assert await checkpoint_saver.aget_tuple(
            {
                "configurable": {
                    "thread_id": "test_thread",
                    "thread_ts": "2021-09-01T00:00:00+00:00",
                }
            }
        ) == CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "test_thread",
                    "thread_ts": "2021-09-01T00:00:00+00:00",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2021-09-01T00:00:00+00:00",
                "channel_values": {},
                "channel_versions": defaultdict(None, {}),
                "versions_seen": defaultdict(None, {}),
            },
            parent_config={
                "configurable": {
                    "thread_id": "test_thread",
                    "thread_ts": "2021-09-01T00:00:00+00:00",
                }
            },
        )
