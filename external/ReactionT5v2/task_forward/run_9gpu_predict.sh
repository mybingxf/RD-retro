#!/usr/bin/env bash
set -euo pipefail

# ====== 你需要改的 3 个路径（prediction.py 路径按你的实际位置写） ======
INPUT_CSV="/root/autodl-fs/try/train/localretro/input_data.csv"
MODEL="/root/autodl-fs/ReactionT5v2/task_forward/weights/ReactionT5v2-forward"
PRED_PY="prediction.py"   # 如果不在当前目录，写绝对路径，比如 /root/autodl-fs/ReactionT5v2/task_forward/prediction.py

# ====== 推理参数 ======
N_GPUS=9
INPUT_MAX_LENGTH=150
NUM_BEAMS=5
NUM_RETURN_SEQS=5
BATCH_SIZE=16

# ====== 新的工作目录（会在这里生成 chunks/ parts/ merged_output.csv） ======
WORKDIR="/root/autodl-fs/try/train/localretro/predict_9gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORKDIR/chunks" "$WORKDIR/parts"

echo "[INFO] Workdir: $WORKDIR"

# ====== 1) 切分 CSV 为 9 份（保持原顺序） ======
export INPUT_CSV WORKDIR N_GPUS
python - <<'PY'
import os
import pandas as pd

input_csv = os.environ["INPUT_CSV"]
workdir = os.environ["WORKDIR"]
n = int(os.environ["N_GPUS"])

df = pd.read_csv(input_csv)
L = len(df)

# 均匀切分：前 rem 份多 1 行
base = L // n
rem = L % n

starts = [0]
for i in range(n):
    size = base + (1 if i < rem else 0)
    starts.append(starts[-1] + size)

for i in range(n):
    s, e = starts[i], starts[i+1]
    out = os.path.join(workdir, "chunks", f"input_part_{i}.csv")
    df.iloc[s:e].to_csv(out, index=False)
    print(f"[SPLIT] part {i}: rows {s}..{e-1} (n={e-s}) -> {out}")

print(f"[SPLIT] total rows = {L}")
PY

# ====== 2) 9 张卡并行推理 ======
for i in $(seq 0 $((N_GPUS-1))); do
  out_dir="$WORKDIR/parts/part_${i}"
  mkdir -p "$out_dir"

  echo "[RUN] GPU $i -> $out_dir"

  CUDA_VISIBLE_DEVICES="$i" \
  python "$PRED_PY" \
    --input_data="$WORKDIR/chunks/input_part_${i}.csv" \
    --model_name_or_path="$MODEL" \
    --input_max_length="$INPUT_MAX_LENGTH" \
    --num_beams="$NUM_BEAMS" \
    --num_return_sequences="$NUM_RETURN_SEQS" \
    --batch_size="$BATCH_SIZE" \
    --output_dir="$out_dir" \
    >"$out_dir/run.log" 2>&1 &
done

wait
echo "[INFO] All shards finished."

# ====== 3) 按顺序合并 9 份 output.csv ======
MERGED="$WORKDIR/merged_output.csv"

# 用 part_0 的表头
head -n 1 "$WORKDIR/parts/part_0/output.csv" > "$MERGED"

# 依次拼接每个分片（跳过表头）
for i in $(seq 0 $((N_GPUS-1))); do
  tail -n +2 "$WORKDIR/parts/part_${i}/output.csv" >> "$MERGED"
done

echo "[OK] Merged file: $MERGED"
echo "[OK] Logs are in: $WORKDIR/parts/part_*/run.log"
