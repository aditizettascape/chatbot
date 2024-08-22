"""GridGain Document Loader for LangChain."""
from typing import Any, Dict, List, Optional, Union
import logging

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from pyignite import Client
from pyignite.exceptions import CacheError, SocketError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IgniteDocumentLoader(BaseLoader):
    """Load documents from Apache Ignite cache."""

    def __init__(
        self,
        cache_name: str,
        ignite_host: str = "127.0.0.1",
        ignite_port: int = 10800,
        filter_criteria: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        create_cache_if_not_exists: bool = True
    ) -> None:
        """
        Initialize the IgniteDocumentLoader.

        Args:
            cache_name: name of the Ignite cache to use.
            ignite_host: Ignite server address.
            ignite_port: Ignite server port.
            filter_criteria: Criteria to filter documents (simple equality checks only).
            limit: a maximum number of documents to return in the read query.
            create_cache_if_not_exists: If True, create the cache if it doesn't exist.
        """
        self.cache_name = cache_name
        self.ignite_host = ignite_host
        self.ignite_port = ignite_port
        self.filter = filter_criteria
        self.limit = limit
        self.create_cache_if_not_exists = create_cache_if_not_exists
        self.client = Client()

    def list_caches(self) -> List[str]:
        """List all available caches in the Ignite cluster."""
        try:
            self.client.connect(self.ignite_host, self.ignite_port)
            return self.client.get_cache_names()
        except SocketError as e:
            logger.error(f"Failed to connect to Ignite server: {e}")
            raise
        finally:
            self._close_client()

    def create_cache(self) -> None:
        """Create the specified cache if it doesn't exist."""
        try:
            self.client.connect(self.ignite_host, self.ignite_port)
            if self.cache_name not in self.client.get_cache_names():
                self.client.create_cache(self.cache_name)
                logger.info(f"Cache '{self.cache_name}' created successfully.")
            else:
                logger.info(f"Cache '{self.cache_name}' already exists.")
        except (SocketError, CacheError) as e:
            logger.error(f"Failed to create cache: {e}")
            raise
        finally:
            self._close_client()
    
    def populate_cache(self, reviews):
        """Populate the cache with sample items."""
        try:
            self.client.connect(self.ignite_host, self.ignite_port)
            cache = self.client.get_or_create_cache(self.cache_name)
        
            for laptop_name, review_text in reviews.items():
                cache.put(laptop_name, review_text)
            logger.info(f"Populated cache '{self.cache_name}' with sample items.")
        except (SocketError, CacheError) as e:
            logger.error(f"Failed to populate cache: {e}")
            raise
        finally:
            self._close_client()
    
    def get(self, key):
        """Retrieve a value from the cache by key."""
        try:
            self.client.connect(self.ignite_host, self.ignite_port)
            cache = self.client.get_or_create_cache(self.cache_name)
            return cache.get(key)
        except (SocketError, CacheError) as e:
            logger.error(f"Failed to get value from cache: {e}")
            raise
        finally:
            self._close_client()

    def _matches_filter(self, value: Union[List, Dict]) -> bool:
        """
        Check if a value matches the filter criteria.
        
        Args:
            value: The value to check against the filter criteria.
        
        Returns:
            bool: True if the value matches the filter, False otherwise.
        """
        if not self.filter:
            return True
        if isinstance(value, list):
            return all(self.filter.get(str(i)) == v for i, v in enumerate(value) if str(i) in self.filter)
        elif isinstance(value, dict):
            return all(value.get(k) == v for k, v in self.filter.items())
        else:
            return False

    def load(self) -> List[Document]:
        """Load data into Document objects."""
        documents = []
        try:
            self.client.connect(self.ignite_host, self.ignite_port)
            cache = self.client.get_cache(self.cache_name)
            
            for key, value in cache.scan():
                documents.append(Document(
                    page_content=str(value),
                    metadata={"key": key, "cache": self.cache_name}
                ))
                if self.limit and len(documents) >= self.limit:
                    break

            return documents

        except (SocketError, CacheError) as e:
            logger.error(f"Failed to load documents: {e}")
            logger.info("Available caches: %s", ", ".join(self.client.get_cache_names()))
            raise
        finally:
            self._close_client()

    def _close_client(self):
        """Close the Ignite client connection."""
        try:
            self.client.close()
        except Exception as e:
            logger.error(f"Failed to close Ignite client connection: {e}")
