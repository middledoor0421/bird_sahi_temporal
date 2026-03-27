#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
WEIGHTS="${WEIGHTS:-weights/yolov8n.pt}"
OUT_ROOT="${OUT_ROOT:-output/thesis2/aligned_verify_baselines/experiments}"
FBD_ROOT="${FBD_ROOT:-/mnt/ssd960/jm/sahi/data/FBD-SV-2024}"
ICCT_ROOT="${ICCT_ROOT:-/mnt/ssd960/jm/sahi/data/ICCT_SPARSE20}"
FAR_SMALL_JSON="${FAR_SMALL_JSON:-output/thesis2/analysis/2026-03-11_far_small_fbdsv_val_tau002.json}"
KEY_INTERVAL="${KEY_INTERVAL:-4}"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <tag> <method>" >&2
  echo "  method: yolo_sahi_always_verify | yolo_sahi_keyframe_verify" >&2
  exit 2
fi

TAG="$1"
METHOD="$2"

case "${METHOD}" in
  yolo_sahi_always_verify)
    PRED_STEM="yolo_sahi_always_verify_exp0"
    ;;
  yolo_sahi_keyframe_verify)
    PRED_STEM="yolo_sahi_keyframe_verify_exp0_n${KEY_INTERVAL}"
    ;;
  *)
    echo "unsupported method: ${METHOD}" >&2
    exit 2
    ;;
esac

run_pair() {
  local dataset="$1"
  local data_root="$2"
  local split="$3"
  local tag="$4"
  local force_one_class="$5"
  local gt_json="$6"
  local seq_json="$7"

  local infer_dir="${OUT_ROOT}/${tag}/infer"
  local kpi_dir="${OUT_ROOT}/${tag}/kpi"
  local pred_path="${infer_dir}/pred_slim_${PRED_STEM}_${split}.jsonl.gz"
  local stats_path="${infer_dir}/stats_${PRED_STEM}_${split}.json"
  local summary_path="${kpi_dir}/summary.json"

  mkdir -p "${infer_dir}" "${kpi_dir}"

  if [[ ! -f "${pred_path}" || ! -f "${stats_path}" ]]; then
    "${PYTHON_BIN}" scripts2/run_detector2.py \
      --datasets "${dataset}" \
      --data-root "${data_root}" \
      --split "${split}" \
      --method "${METHOD}" \
      --weights "${WEIGHTS}" \
      --device "${DEVICE}" \
      --img-size 640 \
      --conf-thres 0.25 \
      --slice-height 512 \
      --slice-width 512 \
      --overlap-height-ratio 0.2 \
      --overlap-width-ratio 0.2 \
      --nms-iou-threshold 0.5 \
      --key-interval "${KEY_INTERVAL}" \
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
      --use-confirmation 1 \
      --use-dual-budget 1 \
      --use-targeted-verify 1
  fi

  if [[ ! -f "${summary_path}" ]]; then
    "${PYTHON_BIN}" scripts2/eval_run3.py \
      --gt "${gt_json}" \
      --seq "${seq_json}" \
      --pred "${pred_path}" \
      --stats "${stats_path}" \
      --method "${METHOD}" \
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
      --key-interval "${KEY_INTERVAL}" \
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
  "${FBD_ROOT}/annotations/fbdsv24_vid_val.json" "${FBD_ROOT}/annotations/fbdsv24_vid_val_sequences.json"

run_pair "icct_sparse20" "${ICCT_ROOT}" "dev" "${TAG}_icct_dev" "1" \
  "${ICCT_ROOT}/annotations/fbdsv24_vid_dev.json" "${ICCT_ROOT}/annotations/fbdsv24_vid_dev_sequences.json"

run_readiness \
  "${OUT_ROOT}/${TAG}_fbdsv_val/infer/pred_slim_${PRED_STEM}_val.jsonl.gz" \
  "${OUT_ROOT}/readiness_${TAG}_fbdsv_val.json"
