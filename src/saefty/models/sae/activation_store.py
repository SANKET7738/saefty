import torch
import numpy as np
from collections import defaultdict
from pydantic import BaseModel
from typing import Optional, List, Iterator, Dict
from datasets import load_dataset, interleave_datasets


class ActivationStoreConfig(BaseModel):
    dataset: str = "CohereLabs/aya_collection_language_split"
    languages: List[str] = ["english", "standard_arabic", "hindi"]
    hook_layer: int = 20
    seq_len: int = 128
    batch_size: int = 4096
    buffer_size: int = 262144
    total_tokens: int = 5_000_000
    seed: int = 42


class ActivationStore:
    def __init__(self, engine, config: ActivationStoreConfig):
        self.engine = engine
        self.config = config
        self.d_model = engine.d_model

        self._rng = np.random.default_rng(config.seed)
        self._buffer = torch.empty(0, self.d_model)
        self._tokens_collected = 0
        self._text_iter = None
        self._lang_token_counts: Dict[str, int] = defaultdict(int)


    def _make_text_iterator(self) -> Iterator[tuple[str, str]]:
        # load per-language subsets and interleave them
        streams = []
        for lang in self.config.languages:
            ds = load_dataset(
                self.config.dataset,
                lang,
                split="train",
                streaming=True,
            )
            streams.append(ds)

        if len(streams) == 1:
            combined = streams[0]
        else:
            combined = interleave_datasets(
                streams, stopping_strategy="all_exhausted",
            )
        combined = combined.shuffle(seed=self.config.seed, buffer_size=10_000)

        for row in combined:
            text = row.get("inputs", "") + " " + row.get("targets", "")
            text = text.strip()
            if len(text) > 20:
                lang = row.get("language", "unknown")
                yield text, lang


    def _collect_activations_from_text(self, text: str) -> torch.Tensor:
        tokens = self.engine.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.seq_len,
        ).to(self.engine.model.device)

        activations = {}
        layer = self.config.hook_layer

        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            activations["act"] = act.detach().cpu().float()

        handle = self.engine.model.model.layers[layer].register_forward_hook(hook_fn)

        with torch.no_grad():
            self.engine.model(**tokens)

        handle.remove()

        # act shape: [1, seq_len, d_model] or [seq_len, d_model]
        act = activations["act"]
        if act.dim() == 3:
            act = act.squeeze(0)
        return act


    def _fill_buffer(self):
        if self._text_iter is None:
            self._text_iter = self._make_text_iterator()

        new_acts = []
        new_count = 0
        target = self.config.buffer_size - len(self._buffer)

        for text, lang in self._text_iter:
            act = self._collect_activations_from_text(text)
            new_acts.append(act)
            n_tokens = act.shape[0]
            new_count += n_tokens
            self._tokens_collected += n_tokens
            self._lang_token_counts[lang] += n_tokens

            if new_count >= target:
                break
            if self._tokens_collected >= self.config.total_tokens:
                break

        if new_acts:
            new_tensor = torch.cat(new_acts, dim=0)
            self._buffer = torch.cat([self._buffer, new_tensor], dim=0)

        # shuffle buffer
        perm = self._rng.permutation(len(self._buffer))
        self._buffer = self._buffer[perm]

        lang_summary = ", ".join(
            f"{lang}: {count:,}" for lang, count in
            sorted(self._lang_token_counts.items(), key=lambda x: -x[1])
        )
        print(f"buffer filled: {len(self._buffer)} activations, "
              f"total tokens seen: {self._tokens_collected:,}")
        print(f"  per-language: {lang_summary}")


    @property
    def language_token_counts(self) -> Dict[str, int]:
        return dict(self._lang_token_counts)


    def __iter__(self) -> Iterator[torch.Tensor]:
        self._tokens_collected = 0
        self._text_iter = None
        self._buffer = torch.empty(0, self.d_model)
        self._lang_token_counts = defaultdict(int)

        while self._tokens_collected < self.config.total_tokens:
            if len(self._buffer) < self.config.batch_size:
                self._fill_buffer()

            if len(self._buffer) < self.config.batch_size:
                # not enough data left
                if len(self._buffer) > 0:
                    yield self._buffer
                break

            batch = self._buffer[:self.config.batch_size]
            self._buffer = self._buffer[self.config.batch_size:]
            yield batch

        print(f"activation store exhausted: {self._tokens_collected:,} tokens total")
        lang_summary = ", ".join(
            f"{lang}: {count:,}" for lang, count in
            sorted(self._lang_token_counts.items(), key=lambda x: -x[1])
        )
        print(f"  final per-language counts: {lang_summary}")
