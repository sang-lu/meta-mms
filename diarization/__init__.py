from importlib import import_module

__all__ = ["MSDDDiarizer", "SortformerDiarizer", "create_diarizer"]


def create_diarizer(name: str, device: str) -> object:
    if name == "msdd":
        module = import_module("diarization.msdd.msdd")
        return module.MSDDDiarizer(device)
    if name == "sortformer":
        module = import_module("diarization.sortformer.sortformer")
        return module.SortformerDiarizer(device)
    raise ValueError(f"Unknown diarizer: {name}")


def __getattr__(name: str):
    if name == "MSDDDiarizer":
        return import_module("diarization.msdd.msdd").MSDDDiarizer
    if name == "SortformerDiarizer":
        return import_module("diarization.sortformer.sortformer").SortformerDiarizer
    raise AttributeError(name)

