from typing import Dict, List, Set, Tuple
from collections import OrderedDict

from vllm.v1.request import Request

class CacheEntry:
    def __init__(self, num_slots: int):
        self.num_slots = num_slots # Number of slots used by this entry.
        self.num_in_use = 0 # How many input are using this entry, the input can from different requests.

# Assumptions:
# - request.mm_hash[input_id] is the hash value for the image #input_id.

# Questions:
# - where is the code pointer that replace the placeholder with the encoder output?
class EncoderCacheManager:

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        # hash -> CacheEntry
        self.cached: Dict[int, CacheEntry] = OrderedDict()
        # List of [req_id, input_id]
        self.freed: List[Tuple[str, int]] = []

    def has_cache(self, request: Request, input_id: int) -> bool:
        req_id = request.request_id
        return req_id in self.cached and input_id in self.cached[req_id]

    def can_allocate(self, request: Request, input_id: int) -> bool:
        num_tokens = request.get_num_encoder_tokens(input_id)
        return num_tokens <= self.num_free_slots

    def allocate(self, request: Request, input_id: int) -> None:
        req_id = request.request_id
        if req_id not in self.cached:
            self.cached[req_id] = set()
        self.cached[req_id].add(input_id)
        self.num_free_slots -= request.get_num_encoder_tokens(input_id)

    def get_cached_input_ids(self, request: Request) -> Set[int]:
        return self.cached.get(request.request_id, set())

    def free(self, request: Request, input_id: int) -> None:
        req_id = request.request_id
        if req_id not in self.cached:
            return

        self.cached[req_id].discard(input_id)
        if len(self.cached[req_id]) == 0:
            del self.cached[req_id]
        self.num_free_slots += request.get_num_encoder_tokens(input_id)
        self.freed.append((req_id, input_id))

    def get_freed_ids(self) -> List[Tuple[str, int]]:
        freed = self.freed
        self.freed = []
        return freed