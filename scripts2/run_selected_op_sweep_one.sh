#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
WEIGHTS="${WEIGHTS:-weights/yolov8n.pt}"
OUT_ROOT="${OUT_ROOT:-output/thesis2/selected_point_sweeps/experiments}"
FBD_ROOT="${FBD_ROOT:-/mnt/ssd960/jm/sahi/data/FBD-SV-2024}"
ICCT_ROOT="${ICCT_ROOT:-/mnt/ssd960/jm/sahi/data/ICCT_SPARSE20}"
FAR_SMALL_JSON="${FAR_SMALL_JSON:-output/thesis2/analysis/2026-03-11_far_small_fbdsv_val_tau002.json}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <tag>" >&2
  exit 2
fi

TAG="$1"

EMPTY_EXPLORE_RATE="0.10"
POS_EXPLORE_RATE="0.15"
EMPTY_EXPLORE_CAP="2.0"
POS_EXPLORE_CAP="2.0"
POS_MOTION_THR="1000000000"
POS_DIFF_RATIO_THR="1.5"

case "${TAG}" in
  2026-03-16_op_lowbudget)
    EMPTY_EXPLORE_RATE="0.08"
    POS_EXPLORE_RATE="0.12"
    ;;
  2026-03-16_op_baseline)
    ;;
  2026-03-16_op_midplus)
    EMPTY_EXPLORE_RATE="0.12"
    POS_EXPLORE_RATE="0.18"
    ;;
  2026-03-16_op_highbudget)
    EMPTY_EXPLORE_RATE="0.15"
    POS_EXPLORE_RATE="0.22"
    ;;
  *)
    echo "unknown tag: ${TAG}" >&2
    exit 2
    ;;
esac

run_pair() {
  local dataset="$1"
  local data_root="$2"
  local split="$3"
  local tag="$4"
  local force_one_class="$5"
  local pred_name="$6"
  local stats_name="$7"
  local gt_json="$8"
  local seq_json="$9"

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
      --use-confirmation 0 \
      --use-dual-budget 1 \
      --use-targeted-verify 1 \
      --empty-explore-rate "${EMPTY_EXPLORE_RATE}" \
      --empty-explore-cap "${EMPTY_EXPLORE_CAP}" \
      --empty-explore-cost 1.0 \
      --pos-explore-rate "${POS_EXPLORE_RATE}" \
      --pos-explore-cap "${POS_EXPLORE_CAP}" \
      --pos-explore-cost 1.0 \
      --pos-motion-local-peak-thr "${POS_MOTION_THR}" \
      --pos-diff-ratio-thr "${POS_DIFF_RATIO_THR}" \
      --tile-score-mode mean \
      --tile-apply-row-quota 0 \
      --tile-quota-mode top1 \
      --tile-use-memory 0 \
      --tile-memory-ttl 0 \
      --tile-cand-extra 6
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

run_readiness() {
  local pred_path="$1"
  local out_path="$2"
  "${PYTHON_BIN}" scripts2/eval_video_kpi2.py \
    --gt "${FBD_ROOT}/annotations/fbdsv24_vid_val.json" \
    --pred "${pred_path}" \
    --out "${out_path}" \
    --image-ids-json "${FAR_SMALL_JSON}" \
    --pred-score-thr 0.25 \
    --match-iou-thr 0.5 \
    --lock-n 3 \
    --ready-n 5 \
    --small-max-area 1024 \
    --small-track-frac 0.5
}

run_pair "fbdsv" "${FBD_ROOT}" "val" "${TAG}_fbdsv_val" "0" \
  "pred_slim_temporal_sahi_v7_exp0_val.jsonl.gz" "stats_temporal_sahi_v7_exp0_val.json" \
  "${FBD_ROOT}/annotations/fbdsv24_vid_val.json" "${FBD_ROOT}/annotations/fbdsv24_vid_val_sequences.json"

run_pair "icct_sparse20" "${ICCT_ROOT}" "dev" "${TAG}_icct_dev" "1" \
  "pred_slim_temporal_sahi_v7_exp0_dev.jsonl.gz" "stats_temporal_sahi_v7_exp0_dev.json" \
  "${ICCT_ROOT}/annotations/fbdsv24_vid_dev.json" "${ICCT_ROOT}/annotations/fbdsv24_vid_dev_sequences.json"

run_readiness \
  "${OUT_ROOT}/${TAG}_fbdsv_val/infer/pred_slim_temporal_sahi_v7_exp0_val.jsonl.gz" \
  "${OUT_ROOT}/readiness_${TAG}_fbdsv_val.json"
