"""
Shared prompt-formatting helper for consistent chat-template handling across
the pipeline.

Three modes:

- ``raw``: return the prompt as a plain string (no chat template applied).
  Matches the SAE training distribution (xP3 text joined into raw strings).
- ``chat``: apply the model's chat template with its default system preamble.
  For ``CohereLabs/tiny-aya-global`` this injects Cohere's built-in safety
  preamble which has historically inflated measured refusal rates.
- ``chat_no_preamble``: apply the chat template, then strip any system-turn
  block from the rendered string. For Cohere-style templates this removes
  the preamble. For templates without a distinct system block (e.g. Gemma)
  the output is equivalent to ``chat``.

Usage::

    from saefty.models.prompt_format import format_prompt, PROMPT_MODES

    text = format_prompt("Tell me how to bake bread.", tokenizer, mode="chat_no_preamble")
"""

from __future__ import annotations

import re
from typing import Dict, List, Union

PROMPT_MODES = ("raw", "chat", "chat_no_preamble")

PromptLike = Union[str, List[Dict[str, str]]]

# Cohere / Command-R family system turn block.
# Example rendered fragment:
#   <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>... preamble ...<|END_OF_TURN_TOKEN|>
_COHERE_SYSTEM_BLOCK_RE = re.compile(
    r"<\|START_OF_TURN_TOKEN\|>\s*<\|SYSTEM_TOKEN\|>.*?<\|END_OF_TURN_TOKEN\|>",
    flags=re.DOTALL,
)

# Generic fallback: Hugging Face style <|system|>...<|end|>
_GENERIC_SYSTEM_BLOCK_RE = re.compile(
    r"<\|system\|>.*?(?=<\|user\|>|<\|assistant\|>|$)",
    flags=re.DOTALL,
)


def _as_messages(prompt: PromptLike) -> List[Dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def _strip_system_block(rendered: str) -> str:
    """Remove any system-turn block from a rendered chat template.

    Safe no-op for templates without a system turn.
    """
    out = _COHERE_SYSTEM_BLOCK_RE.sub("", rendered)
    out = _GENERIC_SYSTEM_BLOCK_RE.sub("", out)
    return out


def format_prompt(
    prompt: PromptLike,
    tokenizer,
    mode: str = "chat_no_preamble",
) -> str:
    """Render ``prompt`` as a plain string under the requested ``mode``.

    Args:
        prompt: Either a raw string or a list of ``{"role", "content"}`` dicts.
        tokenizer: A Hugging Face tokenizer with ``apply_chat_template``.
        mode: One of :data:`PROMPT_MODES`.

    Returns:
        The rendered prompt string.

    Raises:
        ValueError: If ``mode`` is not in :data:`PROMPT_MODES`.
    """
    if mode not in PROMPT_MODES:
        raise ValueError(f"mode must be one of {PROMPT_MODES}, got {mode!r}")

    if mode == "raw":
        if isinstance(prompt, str):
            return prompt
        return "\n".join(m.get("content", "") for m in prompt)

    messages = _as_messages(prompt)
    prefix_forcing = messages and messages[-1]["role"].lower() == "assistant"
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not prefix_forcing,
        continue_final_message=prefix_forcing,
    )

    if mode == "chat_no_preamble":
        rendered = _strip_system_block(rendered)

    return rendered
