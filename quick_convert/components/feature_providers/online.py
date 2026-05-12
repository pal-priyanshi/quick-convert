from __future__ import annotations

from typing import Any

from quick_convert.data.types import AudioBatch, AudioSample


class OnlineFeatureProvider:
    def __init__(self, key: str, extractor: Any):
        self.key = key
        self.extractor = extractor

    def provide_sample(self, sample: AudioSample) -> dict[str, Any]:
        return {self.key: self.extractor.extract_sample(sample)}

    def provide_batch(self, batch: AudioBatch) -> dict[str, Any]:
        return {self.key: self.extractor.extract_batch(batch)}
