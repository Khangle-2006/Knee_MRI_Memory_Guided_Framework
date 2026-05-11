"""
Config loading utility.

Workflow:
    1. Load a YAML config (default: configs/default.yaml).
    2. Wrap it in a dot-accessible namespace.
    3. argparse overrides are merged on top via `apply_overrides`.
    4. `auto`-sentinel paths are resolved via `resolve_view_paths`.
"""
import os
import yaml
from types import SimpleNamespace


def _to_namespace(d):
    """Recursively convert a nested dict into nested SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(v) for v in d]
    return d


def _to_dict(ns):
    """Inverse of _to_namespace (useful for printing / serialization)."""
    if isinstance(ns, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [_to_dict(v) for v in ns]
    return ns


def load_config(yaml_path=None):
    """
    Load a YAML config. If `yaml_path` is None, fall back to the default
    `configs/default.yaml` located next to this file.
    """
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "default.yaml")
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


def apply_overrides(cfg, overrides):
    """
    Apply a flat dict of `{dotted.key: value}` overrides to the config.
    Only non-None values are applied (so unset argparse flags are ignored).

    Example:
        apply_overrides(cfg, {"train.epochs": 50, "data.view": "axial"})
    """
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        parts = dotted_key.split(".")
        target = cfg
        for p in parts[:-1]:
            target = getattr(target, p)
        setattr(target, parts[-1], value)
    return cfg


def resolve_view_paths(cfg):
    """
    Resolve view-dependent paths after overrides have been applied.

    Any path field whose value equals the sentinel string `"auto"` is
    replaced with a default template that includes `cfg.data.view`:

        paths.checkpoint_path  -> ./best_model_{view}_sota_1.pth
        paths.save_dir         -> ./visualizations_{view}

    This way, a user can either:
        - pass `--checkpoint_path /custom/path.pth` to override, OR
        - leave the YAML value as `auto` and have it filled in based
          on the chosen view (sagittal / coronal / axial).
    """
    view = cfg.data.view

    auto_templates = {
        "paths.checkpoint_path": f"./best_model_{view}_sota_1.pth",
        "paths.save_dir":        f"./visualizations_{view}",
    }

    for dotted_key, default_value in auto_templates.items():
        parts = dotted_key.split(".")
        target = cfg
        for p in parts[:-1]:
            target = getattr(target, p)
        current = getattr(target, parts[-1], None)
        if current == "auto":
            setattr(target, parts[-1], default_value)
    return cfg


def print_config(cfg, title="Configuration"):
    """Pretty-print the resolved configuration."""
    import json
    print(f"\n===== {title} =====")
    print(json.dumps(_to_dict(cfg), indent=2))
    print("=" * (len(title) + 12) + "\n")