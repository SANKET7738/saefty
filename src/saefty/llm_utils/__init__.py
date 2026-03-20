from .base_client import BaseLLMClient

__all__ = ["BaseLLMClient"]


def AnthropicClient(*args, **kwargs):
    """Lazy import — requires `pip install anthropic`."""
    from .anthropic_client import AnthropicClient as _Cls
    return _Cls(*args, **kwargs)
