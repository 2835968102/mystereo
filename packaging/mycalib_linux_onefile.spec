# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(SPECPATH).resolve().parent


def add_if_exists(items, src, dest):
    path = project_root / src
    if path.exists():
        items.append((str(path), dest))


datas = []
for script in (project_root / "stereo_calib" / "scripts").glob("*.py"):
    datas.append((str(script), "stereo_calib/scripts"))
add_if_exists(datas, "stereo_calib/scripts/superpoint_v1.pth", "stereo_calib/scripts")
add_if_exists(datas, "configs", "configs")
add_if_exists(datas, "templates", "templates")

binaries = []
add_if_exists(binaries, "build/bin/run_offline_stereo_ba", "build/bin")
add_if_exists(binaries, "build/bin/run_offline_stereo_ba_kitti", "build/bin")
add_if_exists(binaries, "build/bin/run_stereo_calib", "build/bin")

hiddenimports = [
    "cv2",
    "matplotlib",
    "matplotlib.pyplot",
    "numpy",
    "torch",
    "torch.nn",
    "torch.nn.functional",
]

excludes = [
    "tensorflow",
    "torchaudio",
    "torchvision",
]

a = Analysis(
    [str(project_root / "run_calibrate_once.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="mycalib",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
