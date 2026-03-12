import csv
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Optional

from saefty.data.base import PromptLoader, Benchmark, Split, PromptList


LANG_DIRS = ["ar", "bn", "de", "en", "fr", "hi", "ja", "ru", "sp", "zh"]

SKIP_FILES = {"commonsense.csv", "commen_sense.csv"}


class XSafetyConfig(BaseModel):
    data_root: str = "data/xsafety"
    languages: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    take_per_category: Optional[int] = None
    take_per_language: Optional[int] = None


class XSafetyLoader(PromptLoader):
    def __init__(self, config: XSafetyConfig) -> None:
        self.config = config
        self.root = Path(config.data_root)
        if not self.root.exists():
            raise FileNotFoundError(f"xsafety data not found at {self.root}, run: git submodule update --init")
    
    
    def available_languages(self) -> List[str]:
        return [d.name for d in sorted(self.root.iterdir()) if d.is_dir() and d.name in LANG_DIRS]
    
    
    def available_categories(self, language: str = "en") -> List[str]:
        lang_dir = self.root / language
        if not lang_dir.exists():
            return []
        categories = []
        for f in sorted(lang_dir.glob("*.csv")):
            if f.name in SKIP_FILES:
                continue
            categories.append(self._filename_to_category(f.name))
        return categories
    
    
    def _filename_to_category(self, filename: str) -> str:
        name = filename.replace(".csv", "")
        for suffix in ["_en", "_n"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name
    
    
    def _find_csv(self, lang_dir: Path, category: str) -> Optional[Path]:
        candidates = [
            lang_dir / f"{category}.csv",
            lang_dir / f"{category}_en.csv",
            lang_dir / f"{category}_n.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None
    
    
    def _read_csv(self, path: Path) -> List[str]:
        prompts = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[0].strip() if row else ""
                if text:
                    prompts.append(text)
        return prompts
    
    
    def load(self) -> Benchmark:
        languages = self.config.languages or self.available_languages()
        categories = self.config.categories or self.available_categories()
        
        splits: Dict[str, Split] = {}
        
        for lang in languages:
            lang_dir = self.root / lang
            if not lang_dir.exists():
                print(f"warning: language dir not found: {lang}")
                continue
            
            prompts: PromptList = []
            metadata: List[Dict] = []
            
            for category in categories:
                csv_path = self._find_csv(lang_dir, category)
                if csv_path is None:
                    continue
                
                raw_prompts = self._read_csv(csv_path)
                
                if self.config.take_per_category is not None:
                    raw_prompts = raw_prompts[: self.config.take_per_category]
                
                for text in raw_prompts:
                    prompts.append([{"role": "user", "content": text}])
                    metadata.append({"category": category, "is_harmful": True})
            
            splits[lang] = Split(
                name=lang,
                prompts=prompts[: self.config.take_per_language],
                metadata=metadata[: self.config.take_per_language],
            )
        
        total = sum(len(s.prompts) for s in splits.values())
        print(f"loaded {total} prompts across {len(splits)} languages")
        return Benchmark(name="xsafety", splits=splits)
