"""
Shared configuration utilities for SAM 3 project.

This module provides common functions used across all command modules
(predict, train) to avoid code duplication and ensure consistency.

Functions:
    load_yaml_config: Load configuration from YAML file.
    merge_configs: Merge override args into base config (CLI wins).
    get_nested_value: Safely get nested value from config dict.
    to_bool: Convert 'true'/'false' string to bool.
    set_boolean_argument: Add paired --flag/--no-flag CLI arguments.
    config_from_args: Extract config dict from argparse namespace.
    setup_sam3_path: Add the local sam3 submodule to sys.path.

Constants:
    PROJECT_ROOT: Absolute path to the project root directory.
    SAM3_ROOT: Absolute path to the sam3 submodule directory.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# ─── Project root & path setup ──────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAM3_ROOT = PROJECT_ROOT / "sam3"


def setup_sam3_path() -> None:
    """Add the local sam3 submodule to sys.path if it exists.

    Call this at module level in every command script so that
    ``from sam3.model_builder import ...`` resolves to the submodule
    rather than a system-wide install.
    """
    if SAM3_ROOT.exists() and str(SAM3_ROOT) not in sys.path:
        sys.path.insert(0, str(SAM3_ROOT))


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary. Returns empty dict if file is empty.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config else {}


def merge_configs(base_config: Dict, override_args: Dict) -> Dict:
    """Merge override args into base config (deep merge for nested dicts).

    Args:
        base_config: Base configuration dictionary (typically from YAML).
        override_args: Override arguments to merge in (typically from CLI). CLI wins.

    Returns:
        Merged configuration dictionary.
    """
    result = copy.deepcopy(base_config)
    for key, value in override_args.items():
        if value is not None:
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    return result


def get_nested_value(config: Dict, *keys, default=None):
    """Safely get nested value from config dict.

    Args:
        config: Configuration dictionary.
        *keys: Sequence of keys to traverse.
        default: Default value if any key is missing.

    Returns:
        The nested value, or default if any key is missing.
    """
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def to_bool(value: str | bool | None) -> bool | None:
    """Convert 'true'/'false' string or native bool to bool.

    Args:
        value: String or bool value to convert. Case-insensitive for strings.

    Returns:
        True for 'true'/True, False for 'false'/False, None for None or unknown.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def set_boolean_argument(
    parser: argparse.ArgumentParser,
    dest: str,
    flag_name: str | None = None,
    *,
    neg_prefix: str = "no-",
    help_true: str = "",
    help_false: str = "",
) -> None:
    """Add a paired boolean argument (e.g. --compile / --no-compile) to a parser.

    Omitting both flags yields None, allowing YAML defaults to be preserved.
    """
    flag = flag_name or dest.replace("_", "-")

    positive = f"--{flag}"
    negative = f"--{neg_prefix}{flag}"

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        positive,
        dest=dest,
        action="store_const",
        const=True,
        default=None,
        help=help_true or f"Enable {flag}",
    )
    group.add_argument(
        negative,
        dest=dest,
        action="store_const",
        const=False,
        default=None,
        help=help_false or f"Disable {flag}",
    )


def config_from_args(
    args: argparse.Namespace,
    plain: tuple = (),
    boolean: tuple = (),
    rename: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """从 argparse 命名空间提取配置字典。

    Args:
        args: 解析后的命令行参数
        plain: 原样传递的字段名 (直接 getattr, 排除 None)
        boolean: 通过 ``to_bool`` 转换的字段名 (排除 None)
        rename: {arg字段名: config键名} 重映射

    Returns:
        非空字段组成的配置字典
    """
    cfg: Dict[str, Any] = {}
    rename = rename or {}

    for field in plain:
        v = getattr(args, field, None)
        if v is not None:
            cfg[rename.get(field, field)] = v

    for field in boolean:
        v = to_bool(getattr(args, field, None))
        if v is not None:
            cfg[rename.get(field, field)] = v

    return cfg
