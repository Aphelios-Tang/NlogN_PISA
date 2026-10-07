"""Per-layer cache container for the PISA attention adapters."""
class XmixersCache(list):
    @classmethod
    def from_legacy_cache(cls, cache=None):
        return cache if isinstance(cache, cls) else cls([] if cache is None else cache)

    def get_seq_length(self, layer_idx=0):
        return self[layer_idx]["k"].shape[1] if len(self) > layer_idx and self[layer_idx] is not None else 0
