# Auto-loaded by Python at startup if found on PYTHONPATH.
import datasets

_orig = datasets.load_dataset

def load_dataset(*args, **kwargs):
    if kwargs.get("split") == "default":
        kwargs["split"] = "train"
    return _orig(*args, **kwargs)

datasets.load_dataset = load_dataset
