# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import subprocess

from PyInstaller.utils.hooks import collect_submodules


BASE_VERSION = "v1.0"
FALLBACK_BUILD_NUMBER = 29


def _git_build_number() -> int | None:
    repo_root = Path.cwd()
    if not (repo_root / ".git").exists():
        return None
    try:
        out = subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return int(out)
    except Exception:
        return None


build_number = _git_build_number() or FALLBACK_BUILD_NUMBER
exe_name = f"Superconductor fitting {BASE_VERSION} build {build_number}"
hiddenimports = collect_submodules("pyqtgraph")


a = Analysis(
    ["run_fitting.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
