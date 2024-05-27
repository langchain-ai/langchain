import time
import logging
import json
from typing import Sequence, Optional
from enum import Enum
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import OFFSET_END, OFFSET_BEGINNING, Consumer, Producer


logger = logging.getLogger(__name__)

BOOTSTRAP_SERVERS_CONFIG = "bootstrap.servers"

DEFAULT_TTL_MS = 604800000 # 7 days
DEFAULT_REPLICATION_FACTOR = 1
DEFAULT_PARTITION = 3


class ConsumeStartPos(Enum):
    """ Consume start position for Kafka consumer, used to get chat history messages.
        LAST_CONSUMED: Continue from the last consumed offset.
        EARLIEST: Start consuming from the beginning.
        LATEST: Start consuming from the latest offset.
    """
    LAST_CONSUMED = 1
    EARLIEST = 2
    LATEST = 3


def ensure_topic_exists(admin_client: AdminClient,
                        topic_name: str,
                        replication_factor: int,
                        partition: int,
                        ttl_ms: int) -> int:
    """ Create topic if it doesn't exist, and return the number of partitions.
        If the topic already exists, we don't change the topic configuration.
    """

    try:
        topic_metadata = admin_client.list_topics().topics
        if topic_name in topic_metadata:
            num_partitions = len(topic_metadata[topic_name].partitions)
            logger.info(f"Topic {topic_name} already exists with {num_partitions} partitions")
            return num_partitions
    except Exception as e:
        logger.error(f"Failed to list topics: {e}")
        raise e

    topics = [NewTopic(topic_name, num_partitions=partition, replication_factor=replication_factor,
                       config={"retention.ms" : str(ttl_ms)})]
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
    """Chat message history stored in Kafka."""

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
            session_id: The ID for single chat session.
            bootstrap_servers: Comma-separated host/port pairs for establishing initial connection to Kafka cluster.
                               https://kafka.apache.org/documentation.html#adminclientconfigs_bootstrap.servers
            ttl_ms: Time-to-live (milliseconds) for automatic expiration of stored entries.
                    Default 7 days. -1 for no expiration.
                    This config translates to https://kafka.apache.org/documentation.html#topicconfigs_retention.ms
            replication_factor: The replication factor for the topic. Default 1.
            partition: The number of partitions for the topic. Default 3.
        """
        self.session_id = session_id
        self.bootstrap_servers = bootstrap_servers
        self.admin_client = AdminClient({BOOTSTRAP_SERVERS_CONFIG: bootstrap_servers})
        self.num_partitions = ensure_topic_exists(self.admin_client, session_id, replication_factor, partition, ttl_ms)
        self.producer = Producer({BOOTSTRAP_SERVERS_CONFIG: bootstrap_servers})

    def add_messages(self, messages: Sequence[BaseMessage], flush_timeout_seconds: Optional[float] = None) -> None:
        """Add messages to the chat history by producing to the Kafka topic."""

        def delivery_callback(err, msg):
            if err is not None:
                logger.error(f"Message failed delivery: {err}")
            else:
                logger.debug(f"Produced event to topic {msg.topic()}, partition = {msg.partition()}")

        try:
            for message in messages:
                self.producer.produce(topic=self.session_id, value=json.dumps(message_to_dict(message)),
                                      on_delivery=delivery_callback)
            message_remaining = self.producer.flush(flush_timeout_seconds)
            if message_remaining > 0:
                logger.warning(f"{message_remaining} messages are still in-flight.")
        except Exception as e:
            logger.error(f"Failed to add messages to Kafka: {e}")
            raise e

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve the messages for the session, from Kafka topic continuously from last consumed message.
           This method is stateful and maintains consumed(committed) offset based on consumer group.
           To consume from the beginning, use messages_by_pos with ConsumeStartPos.EARLIEST to reset
           position to beginning. To read from latest message, use ConsumeStartPos.LATEST.
        """
        return self.messages_by_pos(consume_start_pos=ConsumeStartPos.LAST_CONSUMED)

    def messages_by_pos(self,
                        consume_start_pos: ConsumeStartPos = ConsumeStartPos.LAST_CONSUMED,
                        max_message_count: Optional[int] = 100,
                        max_time_sec: Optional[float] = 60.0) -> list[BaseMessage]:
        """Retrieve messages from Kafka topic for the session. Please note this method is stateful.
           Internally, this method uses Kafka consumer to consume messages, and maintains the commit offset.

           Args:
                consume_start_pos: Consuming start position for Kafka consumer.
                                   Default is LAST_CONSUMED, which means resuming from last consumed message.
                                   To consume from the beginning, use EARLIEST to reset to beginning and consume.
                                   To consume from the latest message, use LATEST to consume from latest message.
                max_message_count: Maximum number of messages to consume.
                max_time_sec:      Time limit in seconds to consume messages.
          Returns:
                List of BaseMessage objects.
        """
        consumer_config = {
            BOOTSTRAP_SERVERS_CONFIG: self.bootstrap_servers,
            "group.id": self.session_id,
            "auto.offset.reset": "latest" if consume_start_pos == ConsumeStartPos.LATEST else "earliest"
        }

        def assign_beginning(assigned_consumer, assigned_partitions):
            for p in assigned_partitions:
                p.offset = OFFSET_BEGINNING
            assigned_consumer.assign(assigned_partitions)

        def assign_end(assigned_consumer, assigned_partitions):
            for p in assigned_partitions:
                p.offset = OFFSET_END
            assigned_consumer.assign(assigned_partitions)

        assign_callback = None
        if consume_start_pos == ConsumeStartPos.EARLIEST:
            assign_callback = assign_beginning
        elif consume_start_pos == ConsumeStartPos.LATEST:
            assign_callback = assign_end

        messages = []
        with Consumer(consumer_config) as consumer:
            consumer.subscribe([self.session_id], on_assign= assign_callback)
            start_time_sec = time.time()
            while True:
                if max_time_sec is not None and time.time() - start_time_sec > max_time_sec:
                    break
                if max_message_count is not None and len(messages) >= max_message_count:
                    break

                message = consumer.poll(timeout=1.0)
                if message is None: # poll timeout
                    continue
                if message.error() is not None: # error
                    logger.error(f"Consumer error: {message.error()}")
                    continue
                if message.value() is None: # empty value
                    logger.warning("Empty message value")
                    continue
                messages.append(json.loads(message.value()))

        return messages_from_dict(messages)

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

    def close(self):
        """Close the resources."""
        pass
