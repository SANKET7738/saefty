from abc import ABC, abstractmethod
from typing import Any


class IEvaluator(ABC):
    @abstractmethod
    def evaluate(self, response: str, **kwargs) -> Any: ...
