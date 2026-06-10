#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for bin in \
  build/bin/run_offline_stereo_ba \
  build/bin/run_offline_stereo_ba_kitti \
  build/bin/run_stereo_calib
do
  if [[ ! -x "$bin" ]]; then
    echo "Missing $bin"
    echo "Build the C++ targets first:"
    echo "  cmake -S stereo_calib -B build -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build build -j\"\$(nproc)\""
    exit 1
  fi
done

if [[ ! -f stereo_calib/scripts/superpoint_v1.pth ]]; then
  echo "Missing stereo_calib/scripts/superpoint_v1.pth"
  exit 1
fi

if ! python3 -m PyInstaller --version >/dev/null 2>&1; then
  echo "PyInstaller is not installed in this Python environment."
  echo "Install it with:"
  echo "  python3 -m pip install pyinstaller"
  exit 1
fi

python3 -m PyInstaller --clean --noconfirm packaging/mycalib_linux_onefile.spec

echo
echo "Built: dist/mycalib"
