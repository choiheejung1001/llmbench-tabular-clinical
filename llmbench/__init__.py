"""llmbench: serialization, LoRA fine-tuning/inference helpers and evaluation metrics for the LLM vs ML tabular benchmark."""
import os

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG_DIR = os.path.join(PKG_ROOT, "configs")
MODELS = ["qwen2.5-3B", "qwen2.5-7B", "llama3.1-8B", "qwen2.5-14B"]   # manuscript table order


def load_config(name):
    """Load configs/<name>.json."""
    import json
    with open(os.path.join(CONFIG_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)
