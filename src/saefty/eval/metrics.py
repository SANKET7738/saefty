from typing import Dict, List, Any

from saefty.data.base import Benchmark
from saefty.eval.base import IEvaluator


class MetricsResult:
    def __init__(
        self,
        per_split: Dict[str, Dict[str, Any]],
        overall: Dict[str, Any],
    ) -> None:
        self.per_split = per_split
        self.overall = overall
    
    
    def __repr__(self) -> str:
        lines = [f"overall: {self.overall}"]
        for name, metrics in self.per_split.items():
            lines.append(f"  {name}: {metrics}")
        return "\n".join(lines)


class Metrics:
    def __init__(self, evaluator: IEvaluator) -> None:
        self.evaluator = evaluator
    
    
    def compute(self, predictions: List[str], **kwargs) -> Dict[str, Any]:
        results = [self.evaluator.evaluate(pred, **kwargs) for pred in predictions]
        return self._aggregate(results)
    
    
    def compute_benchmark(
        self,
        benchmark: Benchmark,
        predictions: Dict[str, List[str]],
    ) -> MetricsResult:
        per_split: Dict[str, Dict[str, Any]] = {}
        all_preds: List[str] = []
        
        for split_name, split in benchmark.splits.items():
            split_preds = predictions.get(split_name, [])
            
            if len(split_preds) != len(split.prompts):
                print(f"warning: {split_name} has {len(split.prompts)} prompts but {len(split_preds)} predictions")
            
            per_split[split_name] = self.compute(split_preds, lang=split_name)
            all_preds.extend(split_preds)
        
        overall = self.compute(all_preds)
        return MetricsResult(per_split=per_split, overall=overall)
    
    
    def _aggregate(self, results: List[Any]) -> Dict[str, Any]:
        if not results:
            return {"count": 0, "rate": 0.0}
        
        if all(isinstance(r, bool) for r in results):
            count = sum(results)
            total = len(results)
            return {
                "count": count,
                "total": total,
                "rate": count / total,
            }
        
        if all(isinstance(r, (int, float)) for r in results):
            return {
                "total": len(results),
                "mean": sum(results) / len(results),
                "min": min(results),
                "max": max(results),
            }
        
        return {"total": len(results)}
