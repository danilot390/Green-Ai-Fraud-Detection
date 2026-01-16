from pathlib import Path
import yaml

from src.models.cnn_ie import CNNModel
from src.models.conventional_model import ConventionalNN
from src.models.hybrid_model import HybridModel
from src.models.snn_model import SNNModel
from models.lstm_model_ie import LSTMModel
from models.transformer_model_ie import TransformerModel
from src.models.xbost_cnn_bilstm import XBoost_CNN_BiLSTM


MODEL_REGISTRY = {
    "conventional": ConventionalNN,
    "hybrid": HybridModel,
    "snn": SNNModel,
    "lstm": LSTMModel,
    "ileberi_cnn": CNNModel,
    "transformer": TransformerModel,
    "xboost_cnn_bilstm": XBoost_CNN_BiLSTM,
}

_REGISTRY_PATH = Path("models/registry.yaml")


class ModelRegistry:
    """
    Central registry for promoted model artifacts.

    - Enforces valid model types from MODEL_REGISTRY
    - Tracks available versions
    - Provides safe access to active artifacts
    """

    def __init__(self, registry_path: Path | str = _REGISTRY_PATH):
        self.registry_path = Path(registry_path)

        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Model registry not found at: {self.registry_path.resolve()}"
            )

        self._valid_model_types = set(MODEL_REGISTRY.keys())
        self._data = self._load()
        self._validate_registry()

    def _load(self) -> dict:
        with open(self.registry_path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(
                f"Registry file is empty or invalid YAML: {self.registry_path}"
            )

        return data

    def _validate_registry(self):
        """
        Ensure registry.yaml model types match MODEL_REGISTRY.
        """
        models_section = self._data.get("models", {})
        active_section = self._data.get("active", {})

        unknown_models = (
            set(models_section.keys()) | set(active_section.keys())
        ) - self._valid_model_types

        if unknown_models:
            raise ValueError(
                f"Unknown model types in registry.yaml: {unknown_models}. "
                f"Valid types: {sorted(self._valid_model_types)}"
            )

    def reload(self):
        """Reload registry from disk."""
        self._data = self._load()
        self._validate_registry()

    def get_active(self, model_type: str) -> Path:
        """
        Return the active model artifact path for a given model type.
        """
        if model_type not in self._valid_model_types:
            raise ValueError(
                f"Invalid model type '{model_type}'. "
                f"Valid types: {sorted(self._valid_model_types)}"
            )

        try:
            version = self._data["active"][model_type]
            model_entry = self._data["models"][model_type][version]
        except KeyError as e:
            raise KeyError(
                f"No active model registered for type '{model_type}'"
            ) from e

        path = Path(model_entry["path"]).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Registered model artifact not found at: {path}"
            )

        return path

    def list_models(self, model_type: str) -> dict:
        """
        List all registered versions for a given model type.
        """
        if model_type not in self._valid_model_types:
            raise ValueError(
                f"Invalid model type '{model_type}'. "
                f"Valid types: {sorted(self._valid_model_types)}"
            )

        return self._data.get("models", {}).get(model_type, {})

    def get_metadata(self, model_type: str, version: str) -> dict:
        """
        Return metadata for a specific model version.
        """
        if model_type not in self._valid_model_types:
            raise ValueError(
                f"Invalid model type '{model_type}'. "
                f"Valid types: {sorted(self._valid_model_types)}"
            )

        try:
            return self._data["models"][model_type][version]
        except KeyError as e:
            raise KeyError(
                f"Metadata not found for model '{model_type}', version '{version}'"
            ) from e
