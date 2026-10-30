"""Shared fixtures for the fitting tests.

Synthetic TDMS file builders are in tdms_builders.py (import it
directly: ``import tdms_builders as tb``). This file only has fixtures.
"""

from __future__ import annotations

import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(TESTS_DIR), str(TESTS_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tdms_builders as tb  # noqa: E402


@pytest.fixture
def rng(request) -> np.random.Generator:
    """Random generator with a stable seed per test."""
    return np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


@pytest.fixture
def write_tdms(tmp_path):
    """write_tdms(builder, name="test.tdms", index=False, cut=0) -> path."""

    def _write(builder: tb.TdmsBuilder, name: str = "test.tdms", index=False, cut: int = 0) -> str:
        return builder.write(tmp_path / name, index=index, cut=cut)

    return _write


@pytest.fixture
def scenario(tmp_path):
    """scenario(name, seed=1234, index=False) -> (path, builder, notes).

    Names: tdms_builders.ALL_SCENARIOS.
    """

    def _build(name: str, seed: int = 1234, index=False):
        d = tmp_path / f"{name}_{seed}"
        d.mkdir(exist_ok=True)
        return tb.build_scenario(name, d, seed=seed, index=index)

    return _build


@pytest.fixture(params=["pread", "seek"])
def read_path(request, monkeypatch) -> str:
    """Run a test with both positional read paths.

    "pread" is the POSIX path. "seek" is the Windows path (locked seek + read):
    Linux has pread, so the test forces it here.
    """
    if request.param == "seek":
        from fitting import tdms_fastread as fastread

        monkeypatch.setattr(fastread, "_readinto_at", fastread._seek_readinto)
        monkeypatch.setattr(fastread, "pread", fastread._seek_pread)
    return request.param
