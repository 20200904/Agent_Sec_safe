from __future__ import annotations

import contextlib
import os
from typing import List

# Directory holding the vendored upstream module AND its scenario .jsonl files.
UPSTREAM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "upstream")

# RunConfig.dataset -> upstream ScenarioLoader class name.
_DATASET_LOADER = {
    "MedQA": "ScenarioLoaderMedQA",
    "MedQA_Ext": "ScenarioLoaderMedQAExtended",
    "NEJM": "ScenarioLoaderNEJM",
    "NEJM_Ext": "ScenarioLoaderNEJMExtended",
    "MIMICIV": "ScenarioLoaderMIMICIV",
}


@contextlib.contextmanager
def pushd(path: str):
    """Temporarily change the working directory (upstream loaders open relative paths)."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def make_loader(dataset: str):
    """Instantiate the upstream scenario loader for ``dataset`` (from UPSTREAM_DIR)."""
    import upstream.agentclinic as ac
    name = _DATASET_LOADER.get(dataset)
    if name is None:
        raise ValueError("Unknown dataset: {}".format(dataset))
    cls = getattr(ac, name)
    with pushd(UPSTREAM_DIR):
        return cls()


def select_indices(cfg, num_available: int) -> List[int]:
    """Resolve which scenario indices to run.

    ``case_ids`` (explicit subset, preserved order) overrides ``n_scenarios``;
    ``n_scenarios`` takes the contiguous prefix ``0..min(n, N)-1`` exactly like
    upstream ``main``; ``None`` runs all. Deterministic and reproducible.
    """
    if cfg.case_ids is not None:
        out = []
        for cid in cfg.case_ids:
            if not (0 <= cid < num_available):
                raise IndexError("case_id {} out of range [0,{})".format(cid, num_available))
            out.append(cid)
        return out
    n = cfg.n_scenarios
    if n is None:
        n = num_available
    return list(range(0, min(n, num_available)))


def load_scenarios(cfg):
    """Return ``(scenarios, indices)`` for a RunConfig — the selected scenario objects."""
    loader = make_loader(cfg.dataset)
    indices = select_indices(cfg, loader.num_scenarios)
    scenarios = [loader.get_scenario(id=i) for i in indices]
    return scenarios, indices
