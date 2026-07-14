from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

DATASET_REGISTRY = {}

def register_dataset(name: str):
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataset_handler(name: str):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]

class DatasetHandler(ABC):
    @abstractmethod
    def load_dataset(self, config: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def doc_to_text(self, doc: Dict[str, Any]) -> tuple:
        pass

    @abstractmethod
    def extract_answer(self, generated_text: str, doc: Dict[str, Any]) -> str:
        pass

    def evaluate(self, outputs: List[Dict[str, Any]], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None

    def preprocess_generated_text(self, generated_text: str, doc: Dict[str, Any]) -> str:
        return generated_text