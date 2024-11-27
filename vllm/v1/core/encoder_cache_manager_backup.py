from typing import Dict, List, Set, Tuple

from vllm.v1.request import Request


class EncoderCacheManager:

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        # req_id -> cached input ids
        self.cached: Dict[str, Set[int]] = {}
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


from collections import OrderedDict
from typing import Dict, Set, List, Tuple

class CacheEntry:
    def __init__(self, num_tokens: int):
        self.num_tokens = num_tokens
        self.in_use = 0  # Reference count for how many requests are using this cache entry

class EncoderCacheManager:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size  # Total capacity in tokens
        self.num_used_tokens = 0      # Tokens currently used
        # OrderedDict to implement LRU cache: mm_hash -> CacheEntry
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        # Mapping from request_id to set of mm_hashes it's using
        self.request_mm_hashes: Dict[str, Set[str]] = {}
        # Mapping from mm_hash to set of input_ids using it
        self.mm_hash_to_input_ids: Dict[str, Set[int]] = {}

    def has_cache(self, request: 'Request', input_id: int) -> bool:
        mm_hash = request.mm_hash[input_id]
        return mm_hash in self.cache

    def can_allocate(self, request: 'Request', input_id: int) -> bool:
        mm_hash = request.mm_hash[input_id]
        num_tokens = request.get_num_encoder_tokens(input_id)
        
        if mm_hash in self.cache:
            # Already cached; can allocate without additional space
            return True
        else:
            # Calculate available tokens, considering only entries not in use (can be evicted)
            available_tokens = self.cache_size - self.num_used_tokens
            reclaimable_tokens = sum(
                entry.num_tokens for entry in self.cache.values() if entry.in_use == 0
            )
            total_available = available_tokens + reclaimable_tokens
            return num_tokens <= total_available

    def allocate(self, request: 'Request', input_id: int) -> None:
        request_id = request.request_id
        mm_hash = request.mm_hash[input_id]
        num_tokens = request.get_num_encoder_tokens(input_id)
        
        if mm_hash in self.cache:
            # Increment in_use count
            entry = self.cache[mm_hash]
            entry.in_use += 1
            # Move to end to mark as recently used
            if entry.in_use == 1:
                self.cache.move_to_end(mm_hash)
        else:
            # Need to make space if necessary by evicting unused entries
            required_tokens = num_tokens - (self.cache_size - self.num_used_tokens)
            if required_tokens > 0:
                # Evict entries not in use
                tokens_reclaimed = 0
                evicted_mm_hashes = []
                for old_mm_hash, entry in list(self.cache.items()):
                    if entry.in_use == 0:
                        tokens_reclaimed += entry.num_tokens
                        self.cache.pop(old_mm_hash)
                        self.mm_hash_to_input_ids.pop(old_mm_hash, None)
                        if tokens_reclaimed >= required_tokens:
                            break
                if tokens_reclaimed < required_tokens:
                    raise MemoryError("Not enough cache space to allocate the requested resource.")
                self.num_used_tokens -= tokens_reclaimed
            
            # Add new entry to cache
            entry = CacheEntry(num_tokens)
            entry.in_use = 1
            self.cache[mm_hash] = entry
            self.num_used_tokens += num_tokens

        # Update request and mm_hash mappings
        self.request_mm_hashes.setdefault(request_id, set()).add(mm_hash)
        self.mm_hash_to_input_ids.setdefault(mm_hash, set()).add(input_id)

    def get_cached_input_ids(self, request: 'Request') -> Set[int]:
        return {
            input_id for input_id, mm_hash in request.mm_hash.items()
            if mm_hash in self.cache
        }

    def free(self, request: 'Request', input_id: int) -> None:
        request_id = request.request_id
        mm_hash = request.mm_hash[input_id]
        
        if mm_hash not in self.cache:
            return
        
        entry = self.cache[mm_hash]
        entry.in_use -= 1

        if entry.in_use == 0:
            # Move to front to mark as least recently used
            self.cache.move_to_end(mm_hash, last=False)

        # Update mm_hash to input_ids mapping
        input_ids = self.mm_hash_to_input_ids.get(mm_hash)
        if input_ids:
            input_ids.discard(input_id)
            if not input_ids:
                del self.mm_hash_to_input_ids[mm_hash]
        
        # Update request to mm_hashes mapping
        mm_hashes = self.request_mm_hashes.get(request_id)
        if mm_hashes:
            mm_hashes.discard(mm_hash)
            if not mm_hashes:
                del self.request_mm_hashes[request_id]

    def get_freed_ids(self) -> List[Tuple[str, int]]:
        # Collect mm_hashes that are no longer in use
        freed = []
        for mm_hash, entry in self.cache.items():
            if entry.in_use == 0:
                freed.append(mm_hash)
        return freed
