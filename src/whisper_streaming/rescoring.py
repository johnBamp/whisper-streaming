from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
from whisper.decoding import DecodingOptions, DecodingResult, DecodingTask, compression_ratio

_LOG10_TO_LN = float(np.log(10.0))
_NON_ALNUM_RE = re.compile(r"[^a-z0-9']+")
_SPACE_RE = re.compile(r"\s+")


def normalize_lm_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = _NON_ALNUM_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", lowered).strip()


def _length_penalty_term(token_count: int, length_penalty: Optional[float]) -> float:
    count = max(1, int(token_count))
    if length_penalty is None:
        return float(count)
    return float(((5.0 + count) / 6.0) ** float(length_penalty))


def baseline_sequence_score(
    sum_logprob: float,
    token_count: int,
    length_penalty: Optional[float],
) -> float:
    return float(sum_logprob) / _length_penalty_term(token_count, length_penalty)


@dataclass
class RescoreCandidate:
    text: str
    tokens: List[int]
    sum_logprob: float
    avg_logprob: float
    baseline_score: float
    lm_score: float = 0.0
    fused_score: float = 0.0


@dataclass(frozen=True)
class RescoreSelectionMeta:
    applied: bool
    changed_from_baseline: bool
    candidate_count: int
    baseline_index: int
    selected_index: int


@dataclass
class NBestDecodeResult:
    candidates: List[RescoreCandidate]
    language: str
    language_probs: Optional[dict[str, float]]
    audio_features: torch.Tensor
    no_speech_prob: float
    temperature: float


class KenLMScorer:
    def __init__(self, lm_path: str) -> None:
        path = Path(lm_path).expanduser()
        if not path.is_file():
            raise RuntimeError(f"LM file does not exist or is not a file: {path}")
        try:
            import kenlm
        except ImportError as exc:
            raise RuntimeError(
                "KenLM dependency is missing. Install with: "
                "python3 -m pip install 'whisper-streaming[lm]' (or pip install kenlm)."
            ) from exc

        self.path = str(path.resolve())
        self._model = kenlm.Model(self.path)

    def score(self, text: str, normalize_text: bool = True) -> float:
        prepared = normalize_lm_text(text) if normalize_text else _SPACE_RE.sub(" ", text.strip())
        if not prepared:
            # Treat empty hypotheses as very unlikely under LM.
            return -1e9
        log10_score = float(self._model.score(prepared, bos=True, eos=True))
        return log10_score * _LOG10_TO_LN


def decode_nbest(
    model: Any,
    mel: torch.Tensor,
    options: DecodingOptions,
    top_n: Optional[int] = None,
) -> NBestDecodeResult:
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)
    if mel.ndim != 3:
        raise RuntimeError(f"Unexpected mel shape for n-best decode: {tuple(mel.shape)}")

    task = DecodingTask(model, options)
    task.decoder.reset()
    tokenizer = task.tokenizer
    n_audio = int(mel.shape[0])

    audio_features = task._get_audio_features(mel)
    tokens = torch.tensor([task.initial_tokens]).repeat(n_audio, 1)

    languages, language_probs = task._detect_language(audio_features, tokens)
    if options.task == "lang_id":
        raise RuntimeError("LM rescoring does not support task=lang_id.")

    tokens = tokens.repeat_interleave(task.n_group, dim=0).to(audio_features.device)
    tokens, sum_logprobs, no_speech_probs = task._main_loop(audio_features, tokens)

    audio_features = audio_features[:: task.n_group]
    no_speech_probs = no_speech_probs[:: task.n_group]
    tokens = tokens.reshape(n_audio, task.n_group, -1)
    sum_logprobs = sum_logprobs.reshape(n_audio, task.n_group)
    grouped_tokens, grouped_sum_logprobs = task.decoder.finalize(tokens, sum_logprobs)

    if n_audio != 1:
        raise RuntimeError("LM rescoring currently expects a single-audio decode segment.")

    candidates: List[RescoreCandidate] = []
    for sequence_tensor, sequence_sum_logprob in zip(grouped_tokens[0], grouped_sum_logprobs[0]):
        sampled = sequence_tensor[task.sample_begin :]
        eot = (sampled == tokenizer.eot).nonzero()
        if eot.numel() > 0:
            sampled = sampled[: eot[0, 0]]
        token_ids = sampled.tolist()
        text = tokenizer.decode(token_ids).strip()
        sum_logprob = float(sequence_sum_logprob)
        avg_logprob = sum_logprob / float(len(token_ids) + 1)
        baseline_score = baseline_sequence_score(sum_logprob, len(token_ids), options.length_penalty)
        candidates.append(
            RescoreCandidate(
                text=text,
                tokens=token_ids,
                sum_logprob=sum_logprob,
                avg_logprob=avg_logprob,
                baseline_score=baseline_score,
            )
        )

    if not candidates:
        raise RuntimeError("No decode candidates were produced for LM rescoring.")

    candidates.sort(key=lambda c: c.baseline_score, reverse=True)
    if top_n is not None and top_n > 0:
        candidates = candidates[:top_n]

    return NBestDecodeResult(
        candidates=candidates,
        language=languages[0],
        language_probs=language_probs[0] if language_probs is not None else None,
        audio_features=audio_features[0],
        no_speech_prob=float(no_speech_probs[0]),
        temperature=float(options.temperature),
    )


def select_best_candidate(
    candidates: Sequence[RescoreCandidate],
    scorer: KenLMScorer,
    alpha: float,
    beta: float,
    normalize_text: bool = True,
) -> tuple[RescoreCandidate, RescoreSelectionMeta]:
    if not candidates:
        raise RuntimeError("No candidates available for LM rescoring.")

    baseline_idx = int(np.argmax([c.baseline_score for c in candidates]))

    selected_idx = baseline_idx
    selected_score = float("-inf")
    for idx, candidate in enumerate(candidates):
        lm_score = float(scorer.score(candidate.text, normalize_text=normalize_text))
        word_count = len(candidate.text.split())
        fused_score = candidate.baseline_score + (float(alpha) * lm_score) + (float(beta) * float(word_count))
        candidate.lm_score = lm_score
        candidate.fused_score = fused_score
        if fused_score > selected_score:
            selected_score = fused_score
            selected_idx = idx

    meta = RescoreSelectionMeta(
        applied=True,
        changed_from_baseline=(selected_idx != baseline_idx),
        candidate_count=len(candidates),
        baseline_index=baseline_idx,
        selected_index=selected_idx,
    )
    return candidates[selected_idx], meta


def rescore_decode_result(
    model: Any,
    mel: torch.Tensor,
    options: DecodingOptions,
    scorer: KenLMScorer,
    alpha: float,
    beta: float,
    top_n: Optional[int] = None,
    normalize_text: bool = True,
) -> tuple[DecodingResult, RescoreSelectionMeta]:
    nbest = decode_nbest(model=model, mel=mel, options=options, top_n=top_n)
    selected, meta = select_best_candidate(
        candidates=nbest.candidates,
        scorer=scorer,
        alpha=alpha,
        beta=beta,
        normalize_text=normalize_text,
    )

    result = DecodingResult(
        audio_features=nbest.audio_features,
        language=nbest.language,
        language_probs=nbest.language_probs,
        tokens=selected.tokens,
        text=selected.text,
        avg_logprob=selected.avg_logprob,
        no_speech_prob=nbest.no_speech_prob,
        temperature=nbest.temperature,
        compression_ratio=compression_ratio(selected.text),
    )
    return result, meta
