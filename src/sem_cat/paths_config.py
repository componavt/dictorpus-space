from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATHS_CONFIG = Path(__file__).with_name("sem_cat_paths.toml")


@dataclass(frozen=True)
class SemCatPaths:
    """Canonical paths for the semantic categorization pipeline.

    This class holds the resolved Path objects for all core data files:
    - WordNet domains file
    - Concept categories WDH file
    - Concepts catalog (current: 1445 entries)
    - Concept-level WDH output file
    """

    wn_domains: Path
    concept_categories_wdh: Path
    concepts_catalog: Path
    concepts_wdh: Path


def _resolve_repo_path(value: str) -> Path:
    """Resolve a repository-relative path to an absolute Path.

    If the value is an absolute path, it is returned as-is.
    Otherwise, it is resolved relative to the project root.

    Args:
        value: A path string, either absolute or relative to the repo root.

    Returns:
        An absolute Path object.
    """
    p = Path(value)
    return p if p.is_absolute() else (_PROJECT_ROOT / p)


def load_sem_cat_paths(config_path: str | Path | None = None) -> SemCatPaths:
    """Load semantic categorization paths from a TOML config file.

    Args:
        config_path: Optional path to a custom TOML config file.
                     If None, uses src/sem_cat/sem_cat_paths.toml.

    Returns:
        A SemCatPaths instance with resolved Path objects.

    Raises:
        ValueError: If required keys are missing from the config.
        FileNotFoundError: If the config file does not exist.
        tomllib.TOMLDecodeError: If the config file is malformed.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_PATHS_CONFIG
    with path.open("rb") as f:
        raw = tomllib.load(f)

    try:
        section = raw["paths"]
        return SemCatPaths(
            wn_domains=_resolve_repo_path(section["wn_domains"]),
            concept_categories_wdh=_resolve_repo_path(section["concept_categories_wdh"]),
            concepts_catalog=_resolve_repo_path(section["concepts_catalog"]),
            concepts_wdh=_resolve_repo_path(section["concepts_wdh"]),
        )
    except KeyError as e:
        raise ValueError(f"Missing required sem_cat path config key: {e}") from e
