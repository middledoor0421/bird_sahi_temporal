#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-output/thesis2/phase4_candidates/experiments}"
FBD_ROOT="${FBD_ROOT:-/mnt/ssd960/jm/sahi/data/FBD-SV-2024}"
FAR_SMALL_JSON="${FAR_SMALL_JSON:-output/thesis2/analysis/2026-03-11_far_small_fbdsv_val_tau002.json}"

run_readiness() {
  local pred_path="$1"
  local out_path="$2"
  local image_ids_json="${3:-}"

  if [[ -n "${image_ids_json}" ]]; then
    "${PYTHON_BIN}" scripts2/eval_video_kpi2.py \
      --gt "${FBD_ROOT}/annotations/fbdsv24_vid_val.json" \
      --pred "${pred_path}" \
      --out "${out_path}" \
      --image-ids-json "${image_ids_json}" \
      --pred-score-thr 0.25 \
      --match-iou-thr 0.5 \
      --lock-n 3 \
      --ready-n 5 \
      --small-max-area 1024 \
      --small-track-frac 0.5
  else
    "${PYTHON_BIN}" scripts2/eval_video_kpi2.py \
      --gt "${FBD_ROOT}/annotations/fbdsv24_vid_val.json" \
      --pred "${pred_path}" \
      --out "${out_path}" \
      --pred-score-thr 0.25 \
      --match-iou-thr 0.5 \
      --lock-n 3 \
      --ready-n 5 \
      --small-max-area 1024 \
      --small-track-frac 0.5
  fi
}

run_readiness \
  "output/thesis2/experiments/2026-03-08_yolo_fbdsv_val/infer/pred_slim_yolo_val.jsonl.gz" \
  "${OUT_ROOT}/readiness_yolo_fbdsv_val.json" \
  "${FAR_SMALL_JSON}"

run_readiness \
  "output/thesis2/experiments/2026-03-08_fullsahi_fbdsv_val/infer/pred_slim_full_sahi_val.jsonl.gz" \
  "${OUT_ROOT}/readiness_fullsahi_fbdsv_val.json" \
  "${FAR_SMALL_JSON}"

run_readiness \
  "output/thesis2/experiments/2026-03-08_keyframe4_fbdsv_val/infer/pred_slim_keyframe_sahi_n4_val.jsonl.gz" \
  "${OUT_ROOT}/readiness_keyframe4_fbdsv_val.json" \
  "${FAR_SMALL_JSON}"

run_readiness \
  "output/thesis2/experiments/2026-03-08_v7exp0_fbdsv_val/infer/pred_slim_temporal_sahi_v7_exp0_val.jsonl.gz" \
  "${OUT_ROOT}/readiness_v7base_fbdsv_val.json" \
  "${FAR_SMALL_JSON}"

run_readiness \
  "${OUT_ROOT}/2026-03-11_v7_noconfirm_fbdsv_val/infer/pred_slim_temporal_sahi_v7_exp0_val.jsonl.gz" \
  "${OUT_ROOT}/readiness_v7_noconfirm_fbdsv_val.json" \
  "${FAR_SMALL_JSON}"

run_readiness \
  "${OUT_ROOT}/2026-03-11_v7_nodual_fbdsv_val/infer/pred_slim_temporal_sahi_v7_exp0_val.jsonl.gz" \
  "${OUT_ROOT}/readiness_v7_nodual_fbdsv_val.json" \
  "${FAR_SMALL_JSON}"

run_readiness \
  "${OUT_ROOT}/2026-03-11_v7_notargetverify_fbdsv_val/infer/pred_slim_temporal_sahi_v7_exp0_val.jsonl.gz" \
  "${OUT_ROOT}/readiness_v7_notargetverify_fbdsv_val.json" \
  "${FAR_SMALL_JSON}"
