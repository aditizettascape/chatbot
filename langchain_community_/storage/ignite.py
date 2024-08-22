from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.stores import BaseStore
from pygridgain import Client

class GridGainStore(BaseStore[str, str]):
    def __init__(
        self,
        cache_name: str,
        host: str = "127.0.0.1",
        port: int = 10800,
    ) -> None:
        self.client = Client()
        self.client.connect(host, port)
        self.cache = self.client.get_or_create_cache(cache_name)

    def mget(self, keys: Sequence[str]) -> List[Optional[str]]:
        return [self.cache.get(key) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, str]]) -> None:
        for k, v in key_value_pairs:
            self.cache.put(k, v)

    def mdelete(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.cache.remove(key)

    def yield_keys(self, prefix: Optional[str] = None) -> Sequence[str]:
        for entry in self.cache.scan():
            key = entry.key
            if not prefix or key.startswith(prefix):
                yield key
