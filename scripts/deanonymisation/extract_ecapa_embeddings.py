"""Extract ECAPA-TDNN speaker embeddings for a converted set into one .npz file.

Reads the conversion records (``conversions.jsonl``) of one or more conversion
directories, for example the shards of an array run, and embeds either the
anonymised outputs or their original source recordings:

    python scripts/deanonymisation/extract_ecapa_embeddings.py \
        --conversion-dirs outputs/anonymized/knnvc/unversioned/librispeech/fixed-3851-* \
        --split train-clean-100 --audio anonymised --expected 28539 \
        --model-dir /path/to/pretrained_models/spkrec-ecapa-voxceleb \
        --out outputs/embeddings/ecapa/knnvc_fixed3851_train-clean-100_anonymised.npz

The encoder is SpeechBrain's pretrained VoxCeleb ECAPA-TDNN (``spkrec-ecapa-voxceleb``),
loaded with its own ``hyperparams.yaml``. The repository's
``SpeechBrainECAPASpeakerEncoder`` builds a smaller ECAPA variant and cannot
load these weights.

The .npz holds ``embeddings`` [N, 192] float32 plus per-row ``utt_ids``,
``speaker_ids``, ``splits``, ``target_speakers`` and ``audio_paths``. A JSON
file with the same stem records the inputs and the encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset


SAMPLE_RATE = 16_000
EMBEDDING_DIM = 192


def collect_rows(conversion_dirs: list[Path], split: str, audio: str) -> list[dict]:
    """One row per converted utterance of ``split``, sorted by utterance ID."""
    rows, seen = [], {}
    for conversion_dir in conversion_dirs:
        records = conversion_dir / "conversions.jsonl"
        if not records.is_file():
            raise FileNotFoundError(f"No conversion records in {conversion_dir}")
        for line in records.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["split"] != split:
                continue
            utt_id = record["utt_id"]
            if utt_id in seen:
                raise ValueError(f"{utt_id} appears in both {seen[utt_id]} and {conversion_dir}")
            seen[utt_id] = conversion_dir
            assignment = record["assignment"]
            references = assignment.get("references", [])
            targets = sorted({Path(path).parent.parent.name for path in references})
            path = conversion_dir / record["output_audio"] if audio == "anonymised" else Path(assignment["source_path"])
            rows.append(
                {
                    "utt_id": utt_id,
                    # LibriSpeech utterance IDs are <speaker>-<chapter>-<index>.
                    "speaker_id": record["speaker_id"] or utt_id.split("-")[0],
                    "split": split,
                    "target_speaker": "+".join(targets),
                    "audio_path": str(path),
                }
            )
    return sorted(rows, key=lambda row: row["utt_id"])


class AudioFiles(Dataset):
    def __init__(self, paths: list[str], sample_rate: int):
        self.paths, self.sample_rate = paths, sample_rate

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        waveform, sample_rate = sf.read(self.paths[index], dtype="float32", always_2d=True)
        if sample_rate != self.sample_rate:
            raise ValueError(f"{self.paths[index]} is {sample_rate} Hz; ECAPA expects {self.sample_rate} Hz")
        return torch.from_numpy(waveform.mean(axis=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conversion-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--split", required=True, help="source split to keep, e.g. train-clean-100")
    parser.add_argument("--audio", choices=["anonymised", "original"], required=True)
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="local spkrec-ecapa-voxceleb directory with hyperparams.yaml"
    )
    parser.add_argument("--out", type=Path, required=True, help="output .npz path")
    parser.add_argument("--expected", type=int, help="fail unless exactly this many utterances are found")
    parser.add_argument("--limit", type=int, help="embed only the first N utterances (for testing)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=3)
    args = parser.parse_args()

    rows = collect_rows(args.conversion_dirs, args.split, args.audio)
    if args.expected is not None and len(rows) != args.expected:
        raise SystemExit(
            f"Found {len(rows)} utterances of {args.split}, expected {args.expected}. Is a shard unfinished?"
        )
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"{len(rows)} utterances from {len(args.conversion_dirs)} conversion dirs, audio={args.audio}", flush=True)

    from speechbrain.inference.speaker import EncoderClassifier

    model_dir = str(args.model_dir.resolve())
    encoder = EncoderClassifier.from_hparams(source=model_dir, savedir=model_dir, run_opts={"device": args.device})
    encoder.eval()
    loader = DataLoader(
        AudioFiles([row["audio_path"] for row in rows], SAMPLE_RATE),
        batch_size=None,
        num_workers=args.num_workers,
    )
    embeddings = np.empty((len(rows), EMBEDDING_DIM), dtype=np.float32)
    start = time.time()
    for index, waveform in enumerate(loader):
        with torch.inference_mode():
            embedding = encoder.encode_batch(waveform.unsqueeze(0).to(args.device))
        embeddings[index] = embedding.reshape(-1).float().cpu().numpy()
        if (index + 1) % 2000 == 0:
            print(f"{index + 1}/{len(rows)} embedded ({time.time() - start:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(f".{args.out.stem}.partial.npz")
    np.savez(
        partial,
        embeddings=embeddings,
        **{
            key: np.array([row[key[:-1]] for row in rows])
            for key in ["utt_ids", "speaker_ids", "splits", "target_speakers", "audio_paths"]
        },
    )
    os.replace(partial, args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "encoder": "speechbrain-ecapa-voxceleb",
                "model_dir": model_dir,
                "embedding_dim": EMBEDDING_DIM,
                "audio": args.audio,
                "split": args.split,
                "conversion_dirs": [str(path) for path in args.conversion_dirs],
                "utterances": len(rows),
                "speakers": len({row["speaker_id"] for row in rows}),
                "seconds": round(time.time() - start, 1),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.out} ({len(rows)} x {EMBEDDING_DIM}) in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
