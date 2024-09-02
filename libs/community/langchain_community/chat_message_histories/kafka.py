"""Kafka-based chat message history by using confluent-kafka-python.
confluent-kafka-python is under Apache 2.0 license.
https://github.com/confluentinc/confluent-kafka-python
"""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

if TYPE_CHECKING:
    from confluent_kafka import TopicPartition
    from confluent_kafka.admin import AdminClient

logger = logging.getLogger(__name__)

BOOTSTRAP_SERVERS_CONFIG = "bootstrap.servers"

DEFAULT_TTL_MS = 604800000  # 7 days
DEFAULT_REPLICATION_FACTOR = 1
DEFAULT_PARTITION = 3


class ConsumeStartPosition(Enum):
    """Consume start position for Kafka consumer to get chat history messages.
    LAST_CONSUMED: Continue from the last consumed offset.
    EARLIEST: Start consuming from the beginning.
    LATEST: Start consuming from the latest offset.
    """

    LAST_CONSUMED = 1
    EARLIEST = 2
    LATEST = 3


def ensure_topic_exists(
    admin_client: AdminClient,
    topic_name: str,
    replication_factor: int,
    partition: int,
    ttl_ms: int,
) -> int:
    """Create topic if it doesn't exist, and return the number of partitions.
    If the topic already exists, we don't change the topic configuration.
    """
    from confluent_kafka.admin import NewTopic

    try:
        topic_metadata = admin_client.list_topics().topics
        if topic_name in topic_metadata:
            num_partitions = len(topic_metadata[topic_name].partitions)
            logger.info(
                f"Topic {topic_name} already exists with {num_partitions} partitions"
            )
            return num_partitions
    except Exception as e:
        logger.error(f"Failed to list topics: {e}")
        raise e

    topics = [
        NewTopic(
            topic_name,
            num_partitions=partition,
            replication_factor=replication_factor,
            config={"retention.ms": str(ttl_ms)},
        )
    ]
    try:
        futures = admin_client.create_topics(topics)
        for _, f in futures.items():
            f.result()  # result is None
        logger.info(f"Topic {topic_name} created")
    except Exception as e:
        logger.error(f"Failed to create topic {topic_name}: {e}")
        raise e

    return partition


class KafkaChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in Kafka.

    Setup:
        Install ``confluent-kafka-python``.

        .. code-block:: bash

            pip install confluent_kafka

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_message_histories import KafkaChatMessageHistory

            history = KafkaChatMessageHistory(
                session_id="your_session_id",
                bootstrap_servers="host:port",
            )

    Add and retrieve messages:
        .. code-block:: python

            # Add messages
            history.add_messages([message1, message2, message3, ...])

            # Retrieve messages
            message_batch_0 = history.messages

            # retrieve messages after message_batch_0
            message_batch_1 = history.messages

            # Reset to beginning and retrieve messages
            messages_from_beginning = history.messages_from_beginning()

    Retrieving messages is stateful. Internally, it uses Kafka consumer to read.
    The consumed offset is maintained persistently.

    To retrieve messages, you can use the following methods:
    - `messages`:
        continue consuming chat messages from last one.
    - `messages_from_beginning`:
        reset the consumer to the beginning of the chat history and return messages.
        Optional parameters:
        1. `max_message_count`: maximum number of messages to return.
        2. `max_time_sec`: maximum time in seconds to wait for messages.
    - `messages_from_latest`:
        reset to end of the chat history and try consuming messages.
        Optional parameters same as above.
    - `messages_from_last_consumed`:
        continuing from the last consumed message, similar to `messages`.
        Optional parameters same as above.

    `max_message_count` and `max_time_sec` are used to avoid blocking indefinitely
     when retrieving messages. As a result, the method to retrieve messages may not
     return all messages. Change `max_message_count` and `max_time_sec` to retrieve
     all history messages.
    """  # noqa: E501

    def __init__(
        self,
        session_id: str,
        bootstrap_servers: str,
        ttl_ms: int = DEFAULT_TTL_MS,
        replication_factor: int = DEFAULT_REPLICATION_FACTOR,
        partition: int = DEFAULT_PARTITION,
    ):
        """
        Args:
            session_id: The ID for single chat session. It is used as Kafka topic name.
            bootstrap_servers:
                Comma-separated host/port pairs to establish connection to Kafka cluster
                https://kafka.apache.org/documentation.html#adminclientconfigs_bootstrap.servers
            ttl_ms:
                Time-to-live (milliseconds) for automatic expiration of entries.
                Default 7 days. -1 for no expiration.
                It translates to https://kafka.apache.org/documentation.html#topicconfigs_retention.ms
            replication_factor: The replication factor for the topic. Default 1.
            partition: The number of partitions for the topic. Default 3.
        """
        try:
            from confluent_kafka import Producer
            from confluent_kafka.admin import AdminClient
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import confluent_kafka package. "
                "Please install it with `pip install confluent_kafka`."
            )

        self.session_id = session_id
        self.bootstrap_servers = bootstrap_servers
        self.admin_client = AdminClient({BOOTSTRAP_SERVERS_CONFIG: bootstrap_servers})
        self.num_partitions = ensure_topic_exists(
            self.admin_client, session_id, replication_factor, partition, ttl_ms
        )
        self.producer = Producer({BOOTSTRAP_SERVERS_CONFIG: bootstrap_servers})

    def add_messages(
        self,
        messages: Sequence[BaseMessage],
        flush_timeout_seconds: float = 5.0,
    ) -> None:
        """Add messages to the chat history by producing to the Kafka topic."""
        try:
            for message in messages:
                self.producer.produce(
                    topic=self.session_id,
                    value=json.dumps(message_to_dict(message)),
                )
            message_remaining = self.producer.flush(flush_timeout_seconds)
            if message_remaining > 0:
                logger.warning(f"{message_remaining} messages are still in-flight.")
        except Exception as e:
            logger.error(f"Failed to add messages to Kafka: {e}")
            raise e

    def __read_messages(
        self,
        consume_start_pos: ConsumeStartPosition,
        max_message_count: Optional[int],
        max_time_sec: Optional[float],
    ) -> List[BaseMessage]:
        """Retrieve messages from Kafka topic for the session.
           Please note this method is stateful. Internally, it uses Kafka consumer
           to consume messages, and maintains the consumed offset.

         Args:
              consume_start_pos: Start position for Kafka consumer.
              max_message_count: Maximum number of messages to consume.
              max_time_sec:      Time limit in seconds to consume messages.
        Returns:
              List of messages.
        """
        from confluent_kafka import OFFSET_BEGINNING, OFFSET_END, Consumer

        consumer_config = {
            BOOTSTRAP_SERVERS_CONFIG: self.bootstrap_servers,
            "group.id": self.session_id,
            "auto.offset.reset": "latest"
            if consume_start_pos == ConsumeStartPosition.LATEST
            else "earliest",
        }

        def assign_beginning(
            assigned_consumer: Consumer, assigned_partitions: list[TopicPartition]
        ) -> None:
            for p in assigned_partitions:
                p.offset = OFFSET_BEGINNING
            assigned_consumer.assign(assigned_partitions)

        def assign_latest(
            assigned_consumer: Consumer, assigned_partitions: list[TopicPartition]
        ) -> None:
            for p in assigned_partitions:
                p.offset = OFFSET_END
            assigned_consumer.assign(assigned_partitions)

        messages: List[dict] = []
        consumer = Consumer(consumer_config)
        try:
            if consume_start_pos == ConsumeStartPosition.EARLIEST:
                consumer.subscribe([self.session_id], on_assign=assign_beginning)
            elif consume_start_pos == ConsumeStartPosition.LATEST:
                consumer.subscribe([self.session_id], on_assign=assign_latest)
            else:
                consumer.subscribe([self.session_id])
            start_time_sec = time.time()
            while True:
                if (
                    max_time_sec is not None
                    and time.time() - start_time_sec > max_time_sec
                ):
                    break
                if max_message_count is not None and len(messages) >= max_message_count:
                    break

                message = consumer.poll(timeout=1.0)
                if message is None:  # poll timeout
                    continue
                if message.error() is not None:  # error
                    logger.error(f"Consumer error: {message.error()}")
                    continue
                if message.value() is None:  # empty value
                    logger.warning("Empty message value")
                    continue
                messages.append(json.loads(message.value()))
        except Exception as e:
            logger.error(f"Failed to consume messages from Kafka: {e}")
            raise e
        finally:
            consumer.close()

        return messages_from_dict(messages)

    def messages_from_beginning(
        self, max_message_count: Optional[int] = 5, max_time_sec: Optional[float] = 5.0
    ) -> List[BaseMessage]:
        """Retrieve messages from Kafka topic from the beginning.
        This method resets the consumer to the beginning and consumes messages.

             Args:
                 max_message_count: Maximum number of messages to consume.
                 max_time_sec:      Time limit in seconds to consume messages.
             Returns:
                 List of messages.
        """
        return self.__read_messages(
            consume_start_pos=ConsumeStartPosition.EARLIEST,
            max_message_count=max_message_count,
            max_time_sec=max_time_sec,
        )

    def messages_from_latest(
        self, max_message_count: Optional[int] = 5, max_time_sec: Optional[float] = 5.0
    ) -> List[BaseMessage]:
        """Reset to the end offset. Try to consume messages if available.

        Args:
            max_message_count: Maximum number of messages to consume.
            max_time_sec:      Time limit in seconds to consume messages.
        Returns:
            List of messages.
        """

        return self.__read_messages(
            consume_start_pos=ConsumeStartPosition.LATEST,
            max_message_count=max_message_count,
            max_time_sec=max_time_sec,
        )

    def messages_from_last_consumed(
        self, max_message_count: Optional[int] = 5, max_time_sec: Optional[float] = 5.0
    ) -> List[BaseMessage]:
        """Retrieve messages from Kafka topic from the last consumed message.
        Please note this method is stateful. Internally, it uses Kafka consumer
        to consume messages, and maintains the commit offset.

          Args:
               max_message_count: Maximum number of messages to consume.
               max_time_sec:      Time limit in seconds to consume messages.
          Returns:
               List of messages.
        """

        return self.__read_messages(
            consume_start_pos=ConsumeStartPosition.LAST_CONSUMED,
            max_message_count=max_message_count,
            max_time_sec=max_time_sec,
        )

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """
        Retrieve the messages for the session, from Kafka topic continuously
        from last consumed message. This method is stateful and maintains
        consumed(committed) offset based on consumer group.
        Alternatively, use messages_from_last_consumed() with specified parameters.
        Use messages_from_beginning() to read from the earliest message.
        Use messages_from_latest() to read from the latest message.
        """
        return self.messages_from_last_consumed()

    def clear(self) -> None:
        """Clear the chat history by deleting the Kafka topic."""
        try:
            futures = self.admin_client.delete_topics([self.session_id])
            for _, f in futures.items():
                f.result()  # result is None
            logger.info(f"Topic {self.session_id} deleted")
        except Exception as e:
            logger.error(f"Failed to delete topic {self.session_id}: {e}")
            raise e

    def close(self) -> None:
        """Release the resources.
        Nothing to be released at this moment.
        """
        pass
