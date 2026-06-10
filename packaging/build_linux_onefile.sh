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

release_name="mycalib-linux-x86_64-cpu"
release_dir="release"

rm -rf "$release_dir"
mkdir -p "$release_dir"
cp dist/mycalib "$release_dir/$release_name"
chmod +x "$release_dir/$release_name"

if [[ -d packaging/examples ]]; then
  cp -R packaging/examples "$release_dir/examples"
  find "$release_dir/examples" -type f -name "*.sh" -exec chmod +x {} +
fi

tar -C "$release_dir" -czf "$release_dir/$release_name.tar.gz" "$release_name" examples
sha256sum "$release_dir/$release_name" "$release_dir/$release_name.tar.gz" > "$release_dir/SHA256SUMS"

echo
echo "Built: dist/mycalib"
echo "Release package: $release_dir/$release_name.tar.gz"
