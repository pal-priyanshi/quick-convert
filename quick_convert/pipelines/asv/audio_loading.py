from __future__ import annotations

import io
import tarfile
from pathlib import Path

import torchaudio
from speechbrain.dataio import audio_io


def load_audio_from_manifest(
    wav: str,
    *,
    start: int = 0,
    stop: int | None = None,
    shard: str | None = None,
    inner_path: str | None = None,
) -> tuple:
    """Load one audio segment described by an ASV CSV manifest row.

    The original ASV pipeline stores normal file paths in the ``wav`` column and
    SpeechBrain can load those directly. Emilia is stored as tar shards, so its
    manifest rows can point to audio inside an archive using either:

    - ``wav="tar:///path/to/shard.tar::path/inside/shard.mp3"``
    - ``shard="/path/to/shard.tar"`` plus ``inner_path="path/inside/shard.mp3"``

    ``start`` and ``stop`` are frame offsets from the CSV. A negative/missing
    ``stop`` = "read until the end"."""
    start = int(start)
    stop = None if stop is None else int(stop)
    num_frames = -1 if stop is None or stop < 0 else stop - start

    parsed = _parse_tar_uri(wav) # If wav uses tar://SHARD::MEMBER, extract the shard path and inner audio path from it.

    if parsed is not None:
        shard, inner_path = parsed

    if shard and inner_path:
        return _load_from_tar(
            shard=shard,
            inner_path=inner_path,
            frame_offset=start,
            num_frames=num_frames,
        )

    return audio_io.load(wav, num_frames=num_frames, frame_offset=start) # Otherwise, wav is a normal audio file path, so use SpeechBrain's original loader.



def _parse_tar_uri(wav: str) -> tuple[str, str] | None:
    """Split ``tar://SHARD::MEMBER`` into the outer shard and inner file path.

    Returning ``None`` is for when the value is a normal audio path and should be left
    to SpeechBrain's regular loader.
    """
    if not wav.startswith("tar://"):
        return None

    value = wav[len("tar://") :]
    if "::" not in value:
        raise ValueError(f"Expected tar URI in the form tar://SHARD::MEMBER, got {wav}")

    shard, inner_path = value.split("::", 1)
    if not shard or not inner_path:
        raise ValueError(f"Expected tar URI in the form tar://SHARD::MEMBER, got {wav}")

    return shard, inner_path


def _load_from_tar(
    *,
    shard: str,
    inner_path: str,
    frame_offset: int = 0,
    num_frames: int = -1,
) -> tuple:
    """Read an audio file stored inside a tar shard.
    
    The inner audio file is copied into memory as bytes, then passed to
    ``torchaudio.load``. This avoids writing temporary extracted files while
    still returning the same ``(waveform, sample_rate)`` shape expected by the
    rest of the ASV code.
    """
    shard_path = Path(shard)
    if not shard_path.is_file():
        raise FileNotFoundError(f"Emilia shard not found: {shard_path}")

    with tarfile.open(shard_path, "r:*") as tar:
        #Locate the specific audio file inside the tar archive, open it, and 
        #read its raw audio data into memory so torchaudio can decode it.
        member = tar.getmember(inner_path)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Could not read {inner_path} from {shard_path}")

        audio_bytes = io.BytesIO(extracted.read())

    waveform, sample_rate = torchaudio.load(audio_bytes)

    if frame_offset > 0 or num_frames >= 0:
        end = None if num_frames < 0 else frame_offset + num_frames
        waveform = waveform[:, frame_offset:end]

    return waveform, sample_rate