"""Ignite-based chat message history for LangChain."""
from __future__ import annotations

import json
import logging
from typing import List, Optional, Sequence

from pyignite import Client
from pyignite.exceptions import SocketError, CacheError
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CACHE_NAME = "langchain_message_store"

class IgniteChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in Apache Ignite."""

    def __init__(
        self,
        *,
        session_id: str,
        cache_name: str = DEFAULT_CACHE_NAME,
        ignite_host: str = "127.0.0.1",
        ignite_port: int = 10800,
        client: Optional[Client] = None,
    ) -> None:
        """
        Initialize the IgniteChatMessageHistory.

        Args:
            session_id: Arbitrary key used to store the messages of a single chat session.
            cache_name: Name of the Ignite cache to create/use.
            ignite_host: Hostname of the Ignite server.
            ignite_port: Port number of the Ignite server.
            client: Optional pre-configured Ignite client.
        """
        self.session_id = session_id
        self.cache_name = cache_name

        if client:
            self.client = client
        else:
            self.client = Client()
            try:
                self.client.connect(ignite_host, ignite_port)
            except SocketError as e:
                logger.error(f"Failed to connect to Ignite server: {e}")
                raise

        try:
            self.cache = self.client.get_or_create_cache(self.cache_name)
        except CacheError as e:
            logger.error(f"Failed to get or create cache: {e}")
            raise

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all session messages from Ignite."""
        key = f"{self.session_id}_messages"
        try:
            messages_json = self.cache.get(key)
            if not messages_json:
                return []
            items = json.loads(messages_json)
            return messages_from_dict(items)
        except (CacheError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve messages: {e}")
            raise

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        Add messages to the Ignite cache.

        Args:
            messages: Sequence of messages to add.
        """
        key = f"{self.session_id}_messages"
        try:
            existing_messages = self.messages
            new_messages = existing_messages + list(messages)
            messages_json = json.dumps([message_to_dict(m) for m in new_messages])
            self.cache.put(key, messages_json)
        except (CacheError, json.JSONEncodeError) as e:
            logger.error(f"Failed to add messages: {e}")
            raise

    def clear(self) -> None:
        """Clear session messages from the Ignite cache."""
        key = f"{self.session_id}_messages"
        try:
            self.cache.remove_key(key)
        except CacheError as e:
            logger.error(f"Failed to clear messages: {e}")
            raise

    def __del__(self):
        """Close the Ignite client connection when the object is destroyed."""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except Exception as e:
                logger.error(f"Failed to close Ignite client connection: {e}")
