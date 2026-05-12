from pathlib import Path

from ...data.types import AudioSample


class FeatureProvider:
    key: str

    def provide_sample(self, sample: AudioSample) -> dict:
        raise NotImplementedError


class PatternSidecarFeatureProvider:
    def __init__(self, key, root, pattern, load=False, loader=None, **format_kwargs):
        self.key = key
        self.root = Path(root)
        self.pattern = pattern
        self.load = load
        self.loader = loader
        self.format_kwargs = format_kwargs

    def provide_sample(self, sample: AudioSample) -> dict:
        path = self.root / self.pattern.format(
            stem=sample.path.stem,
            name=sample.path.name,
            split=sample.split,
            spk_id=sample.spk_id,
            **self.format_kwargs,
        )

        return {self.key: self.loader(path) if self.load else path}
