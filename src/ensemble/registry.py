from pathlib import Path
import yaml


from src.ensemble.i_s_stacking import IleberiSunStackingModel
from src.ensemble.green_xgboost_stack_model import HybridStackingModel

ENSEMBLE_REGISTRY = {
    'ileberi_sun': IleberiSunStackingModel,
    'hybrid_stacking': HybridStackingModel,
}

_ENSEMBLE_REGISTRY_PATH = Path("ensembles/registry.yaml")


class EnsembleRegistry:
    """
    Central registry for ensemble artifacts.
    """

    def __init__(self, registry_path = _ENSEMBLE_REGISTRY_PATH):
        self.registry_path = Path(registry_path)

        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Ensemble registry not found at: {self.registry_path.resolve()}"
            )

        self._valid_ensembles = set(ENSEMBLE_REGISTRY.keys())
        self._data = self._load()
        self._validate_registry()

    def _load(self) -> dict:
        with open(self.registry_path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(
                f"Ensemble registry is empty or invalid YAML: {self.registry_path}"
            )

        return data

    def _validate_registry(self):
        ensembles_section = self._data.get("ensembles", {})
        active_section = self._data.get("active", {})

        unknown = (
            set(ensembles_section.keys()) | set(active_section.keys())
        ) - self._valid_ensembles

        if unknown:
            raise ValueError(
                f"Unknown ensemble types in registry.yaml: {unknown}. "
                f"Valid types: {sorted(self._valid_ensembles)}"
            )

    def reload(self):
        self._data = self._load()
        self._validate_registry()

    def get_active(self, ensemble_type: str) -> dict:
        """
        Return the active ensemble entry (not just a path).
        """
        if ensemble_type not in self._valid_ensembles:
            raise ValueError(
                f"Invalid ensemble type '{ensemble_type}'. "
                f"Valid types: {sorted(self._valid_ensembles)}"
            )

        try:
            version = self._data["active"][ensemble_type]
            return self._data["ensembles"][ensemble_type][version]
        except KeyError as e:
            raise KeyError(
                f"No active ensemble registered for '{ensemble_type}'"
            ) from e

    def list_ensembles(self, ensemble_type: str) -> dict:
        if ensemble_type not in self._valid_ensembles:
            raise ValueError(
                f"Invalid ensemble type '{ensemble_type}'. "
                f"Valid types: {sorted(self._valid_ensembles)}"
            )

        return self._data.get("ensembles", {}).get(ensemble_type, {})

    def get_metadata(self, ensemble_type: str, version: str) -> dict:
        if ensemble_type not in self._valid_ensembles:
            raise ValueError(
                f"Invalid ensemble type '{ensemble_type}'. "
                f"Valid types: {sorted(self._valid_ensembles)}"
            )

        try:
            return self._data["ensembles"][ensemble_type][version]
        except KeyError as e:
            raise KeyError(
                f"Metadata not found for ensemble '{ensemble_type}', version '{version}'"
            ) from e