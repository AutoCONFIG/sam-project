"""
SAM 3 Unified CLI Entry Point
=============================

Usage:

    python sam.py configs/predict/video_text.yaml
    python sam.py configs/train/roboflow_finetune.yaml

Mode is auto-detected from the 'mode' field in the YAML config file.
CLI flags override YAML values (see commands/<mode>.py for details).
"""

import sys
from pathlib import Path


MODES = {
    "predict": "commands.predict",
    "train": "commands.train",
}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return
    if len(sys.argv) < 2:
        print("Error: expected a YAML config path (plus optional CLI overrides)")
        sys.exit(1)

    import yaml

    config_path = Path(sys.argv[1])
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        print(f"Error: config must be a YAML file: {config_path}")
        sys.exit(1)

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        mode = cfg.get("mode") if isinstance(cfg, dict) else None
    except Exception as e:
        print(f"Error reading config: {e}")
        sys.exit(1)

    if mode not in MODES:
        print(f"Error: invalid or missing 'mode' in config (got: {mode})")
        print(f"Valid modes: {', '.join(MODES.keys())}")
        sys.exit(1)

    # Rewrite sys.argv: ["sam.py", "--config", <path>, ...extra CLI flags...]
    extra_flags = sys.argv[2:]
    sys.argv = ["sam.py", "--config", str(config_path)] + extra_flags
    import importlib
    importlib.import_module(MODES[mode]).main()


if __name__ == "__main__":
    main()
