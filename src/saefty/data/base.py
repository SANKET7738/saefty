from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Dict, Optional


PromptList = List[List[Dict[str, str]]]


class Split(BaseModel):
    name: str 
    prompts: PromptList
    metadata: Optional[List[Dict]] = None


class Benchmark(BaseModel):
    name: str 
    splits: Dict[str, Split]


class PromptLoader(ABC):
    @abstractmethod
    def load(self) -> Benchmark: ...
