#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-roundtrip_demo_run}"
device="${2:-cpu}"
gpu_id="${3:-0}"

python "$package_root/source/run_roundtrip_pipeline.py" \
  --input-csv "$package_root/demo/predictions_demo.csv" \
  --gt-csv "$package_root/demo/aligned_gt_demo.csv" \
  --output-dir "$output_dir" \
  --device "$device" \
  --gpu-id "$gpu_id" \
  --translate-chunk 0

python "$package_root/scripts/validate_demo.py" "$output_dir"
