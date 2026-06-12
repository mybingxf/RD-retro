#!/usr/bin/env bash
set -euo pipefail

# ===== 输入 =====
IN_TXT="/root/autodl-fs/nature/all_data/product_reactants_with_gt_filtered10.txt"
GT_CSV="/root/autodl-fs/localretro/data/USPTO_50K/raw_train.csv"

# ===== worker 脚本路径 =====
WORKER_PY="/root/autodl-fs/nature/all_data/map_txt_to_gtcsv_worker.py"  # 按你实际保存位置改

# ===== 使用 GPU1-8（第2-9张卡）=====
GPU_LIST=(1 2 3 4 5 6 7 8)
N=${#GPU_LIST[@]}

# ===== 输出目录 =====
WORKDIR="/root/autodl-fs/nature/all_data/atommap_txt_8gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORKDIR/chunks" "$WORKDIR/parts" "$WORKDIR/logs"

# ===== 参数 =====
BATCH_SIZE=64

echo "[INFO] WORKDIR : $WORKDIR"
echo "[INFO] IN_TXT  : $IN_TXT"
echo "[INFO] GT_CSV  : $GT_CSV"
echo "[INFO] GPUs    : ${GPU_LIST[*]}"

# 1) 切分 txt（保持顺序）
split -n l/$N -d --additional-suffix=.txt "$IN_TXT" "$WORKDIR/chunks/part_"

# 2) 8 卡并行映射 → 每份一个分片 CSV
for idx in $(seq 0 $((N-1))); do
  part=$(printf "%02d" "$idx")
  gpu=${GPU_LIST[$idx]}

  echo "[RUN] part_$part on GPU$gpu"

  CUDA_VISIBLE_DEVICES="$gpu" \
  python "$WORKER_PY" \
    -i "$WORKDIR/chunks/part_${part}.txt" \
    -o "$WORKDIR/parts/new_${part}.csv" \
    --failed_txt "$WORKDIR/parts/failed_${part}.txt" \
    --batch_size "$BATCH_SIZE" \
    > "$WORKDIR/logs/run_${part}.log" 2>&1 &
done

wait
echo "[INFO] all parts finished."

# 3) 合并新生成的 CSV（只保留一个表头）
NEW_CSV="$WORKDIR/new_mapped_UNK.csv"
head -n 1 "$WORKDIR/parts/new_00.csv" > "$NEW_CSV"
for idx in $(seq 0 $((N-1))); do
  part=$(printf "%02d" "$idx")
  tail -n +2 "$WORKDIR/parts/new_${part}.csv" >> "$NEW_CSV"
done
echo "[OK] new csv: $NEW_CSV"

# 4) 拼接 GT 到新 CSV 后面，得到 merged 训练集（GT 跳过表头）
MERGED_CSV="$WORKDIR/merged_train.csv"
cp "$NEW_CSV" "$MERGED_CSV"
tail -n +2 "$GT_CSV" >> "$MERGED_CSV"

echo "[OK] merged csv: $MERGED_CSV"
echo "[OK] logs: $WORKDIR/logs/run_*.log"
echo "[OK] failed lines (if any): $WORKDIR/parts/failed_*.txt"
