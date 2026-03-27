#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
WEIGHTS="${WEIGHTS:-weights/yolov8n.pt}"
OUT_ROOT="${OUT_ROOT:-output/thesis2/phase4_candidates/experiments}"

run_pair() {
  local dataset="$1"
  local data_root="$2"
  local split="$3"
  local tag="$4"
  local use_confirmation="$5"
  local use_dual_budget="$6"
  local use_targeted_verify="$7"
  local force_one_class="$8"
  local pred_name="$9"
  local stats_name="${10}"
  local gt_json="${11}"
  local seq_json="${12}"

  local infer_dir="${OUT_ROOT}/${tag}/infer"
  local kpi_dir="${OUT_ROOT}/${tag}/kpi"
  local pred_path="${infer_dir}/${pred_name}"
  local stats_path="${infer_dir}/${stats_name}"
  local summary_path="${kpi_dir}/summary.json"

  mkdir -p "${infer_dir}" "${kpi_dir}"

  if [[ ! -f "${pred_path}" || ! -f "${stats_path}" ]]; then
    "${PYTHON_BIN}" scripts2/run_detector2.py \
      --datasets "${dataset}" \
      --data-root "${data_root}" \
      --split "${split}" \
      --method temporal_sahi_v7 \
      --weights "${WEIGHTS}" \
      --device "${DEVICE}" \
      --img-size 640 \
      --conf-thres 0.25 \
      --slice-height 512 \
      --slice-width 512 \
      --overlap-height-ratio 0.2 \
      --overlap-width-ratio 0.2 \
      --nms-iou-threshold 0.5 \
      --out-dir "${infer_dir}" \
      --save-pred-slim 1 \
      --pred-max-det 100 \
      --force-one-class "${force_one_class}" \
      --v7-exp 0 \
      --v7-confirm-iou 0.4 \
      --v7-confirm-len 2 \
      --v7-confirm-max-age 2 \
      --v7-log-path "${infer_dir}/v7_log.json" \
      --v7-cost-per-frame "${infer_dir}/cost_per_frame.jsonl.gz" \
      --use-confirmation "${use_confirmation}" \
      --use-dual-budget "${use_dual_budget}" \
      --use-targeted-verify "${use_targeted_verify}"
  fi

  if [[ ! -f "${summary_path}" ]]; then
    "${PYTHON_BIN}" scripts2/eval_run3.py \
      --gt "${gt_json}" \
      --seq "${seq_json}" \
      --pred "${pred_path}" \
      --stats "${stats_path}" \
      --method temporal_sahi_v7 \
      --dataset "${dataset}" \
      --split "${split}" \
      --exp-id 0 \
      --setting standard \
      --seed 42 \
      --img-size 640 \
      --conf-thr 0.25 \
      --slice-height 512 \
      --slice-width 512 \
      --overlap-height-ratio 0.2 \
      --overlap-width-ratio 0.2 \
      --merge-nms-iou 0.5 \
      --out-dir "${kpi_dir}" \
      --save-per-frame 1
  fi
}

FBD_ROOT="/mnt/ssd960/jm/sahi/data/FBD-SV-2024"
ICCT_ROOT="/mnt/ssd960/jm/sahi/data/ICCT_SPARSE20"

run_pair \
  "fbdsv" \
  "${FBD_ROOT}" \
  "val" \
  "2026-03-11_v7_noconfirm_fbdsv_val" \
  "0" "1" "1" "0" \
  "pred_slim_temporal_sahi_v7_exp0_val.jsonl.gz" \
  "stats_temporal_sahi_v7_exp0_val.json" \
  "${FBD_ROOT}/annotations/fbdsv24_vid_val.json" \
  "${FBD_ROOT}/annotations/fbdsv24_vid_val_sequences.json"

run_pair \
  "fbdsv" \
  "${FBD_ROOT}" \
  "val" \
  "2026-03-11_v7_nodual_fbdsv_val" \
  "1" "0" "1" "0" \
  "pred_slim_temporal_sahi_v7_exp0_val.jsonl.gz" \
  "stats_temporal_sahi_v7_exp0_val.json" \
  "${FBD_ROOT}/annotations/fbdsv24_vid_val.json" \
  "${FBD_ROOT}/annotations/fbdsv24_vid_val_sequences.json"

run_pair \
  "fbdsv" \
  "${FBD_ROOT}" \
  "val" \
  "2026-03-11_v7_notargetverify_fbdsv_val" \
  "1" "1" "0" "0" \
  "pred_slim_temporal_sahi_v7_exp0_val.jsonl.gz" \
  "stats_temporal_sahi_v7_exp0_val.json" \
  "${FBD_ROOT}/annotations/fbdsv24_vid_val.json" \
  "${FBD_ROOT}/annotations/fbdsv24_vid_val_sequences.json"

run_pair \
  "icct_sparse20" \
  "${ICCT_ROOT}" \
  "dev" \
  "2026-03-11_v7_noconfirm_icct_dev" \
  "0" "1" "1" "1" \
  "pred_slim_temporal_sahi_v7_exp0_dev.jsonl.gz" \
  "stats_temporal_sahi_v7_exp0_dev.json" \
  "${ICCT_ROOT}/annotations/fbdsv24_vid_dev.json" \
  "${ICCT_ROOT}/annotations/fbdsv24_vid_dev_sequences.json"

run_pair \
  "icct_sparse20" \
  "${ICCT_ROOT}" \
  "dev" \
  "2026-03-11_v7_nodual_icct_dev" \
  "1" "0" "1" "1" \
  "pred_slim_temporal_sahi_v7_exp0_dev.jsonl.gz" \
  "stats_temporal_sahi_v7_exp0_dev.json" \
  "${ICCT_ROOT}/annotations/fbdsv24_vid_dev.json" \
  "${ICCT_ROOT}/annotations/fbdsv24_vid_dev_sequences.json"

run_pair \
  "icct_sparse20" \
  "${ICCT_ROOT}" \
  "dev" \
  "2026-03-11_v7_notargetverify_icct_dev" \
  "1" "1" "0" "1" \
  "pred_slim_temporal_sahi_v7_exp0_dev.jsonl.gz" \
  "stats_temporal_sahi_v7_exp0_dev.json" \
  "${ICCT_ROOT}/annotations/fbdsv24_vid_dev.json" \
  "${ICCT_ROOT}/annotations/fbdsv24_vid_dev_sequences.json"
