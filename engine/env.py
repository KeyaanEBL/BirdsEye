"""Central path/env config for BirdsEye.

All machine-specific paths live in a `.env` file (simple KEY=VALUE lines) at the
repo root, or wherever $BIRDSEYE_ENV points. A real OS environment variable of
the same name overrides the file. Nothing else in the codebase hardcodes a path.

This module also bridges to the Intern-Project data pipeline — BirdsEye reads raw
/mnt data through it (`data.load_columns`, `config.get_column_map`, …) — using
INTERN_PROJECT_DIR from the .env.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../BirdsEye


def _load_dotenv(path):
    out = {}
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


ENV_PATH = os.environ.get("BIRDSEYE_ENV", os.path.join(_REPO_ROOT, ".env"))


def env(key, default=None):
    """Resolve a setting: OS environment variable > .env file > default.

    The .env file is re-read on every call, so editing it takes effect on the
    next run without re-importing — `run()`/the learners read paths through this.
    """
    if key in os.environ:
        return os.environ[key]
    return _load_dotenv(ENV_PATH).get(key, default)


INTERN_PROJECT_DIR = env("INTERN_PROJECT_DIR")    # required (no hardcoded default)
MANIFEST_PATH      = env("MANIFEST_PATH")          # None if unset
DATA_DIR           = env("DATA_DIR")               # optional raw-dir override (None = config default)
LOG_DIR            = env("LOG_DIR", os.path.join(_REPO_ROOT, "logs"))   # run logs (repo-relative default)


# ---- bridge to the Intern-Project data pipeline ---------------------------
if not INTERN_PROJECT_DIR:
    raise ImportError(
        f"INTERN_PROJECT_DIR is not set. Add it to {ENV_PATH} (or export it), "
        f"pointing at your Intern-Project checkout — BirdsEye reads raw data "
        f"through its pipeline.")

if INTERN_PROJECT_DIR not in sys.path:
    sys.path.insert(0, INTERN_PROJECT_DIR)

try:
    from data import load_columns, get_manifest_files          # noqa: E402
    from config import get_column_map, get_data_dir            # noqa: E402
except Exception as e:                                          # pragma: no cover
    raise ImportError(
        f"BirdsEye reads raw data through the Intern-Project pipeline, but could "
        f"not import it from INTERN_PROJECT_DIR={INTERN_PROJECT_DIR!r} "
        f"(.env at {ENV_PATH}). Fix INTERN_PROJECT_DIR there. Original error: {e}") from e


__all__ = ["env", "ENV_PATH", "INTERN_PROJECT_DIR", "MANIFEST_PATH", "DATA_DIR", "LOG_DIR",
           "load_columns", "get_manifest_files", "get_column_map", "get_data_dir"]
