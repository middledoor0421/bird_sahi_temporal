#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
WEIGHTS="${WEIGHTS:-weights/yolov8n.pt}"

EXP_ROOT="${EXP_ROOT:-output/thesis2/phase4b_support_final/experiments}"
EFFR_ROOT="${EFFR_ROOT:-output/thesis2/phase4b_support_final/target_effr}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <tag>" >&2
  exit 2
fi

TAG="$1"

run_case() {
  local dataset="$1"
  local data_root="$2"
  local split="$3"
  local target_category_id="$4"
  local target_category_name="$5"
  local keep_all_categories="$6"
  local class_mapping_json="$7"
  local empty_mask="$8"
  local effr_group="$9"
  local pred_name="${10}"
  local stats_name="${11}"
  local step2_method_dir="${12}"
  local step3_method_dir="${13}"

  local infer_dir="${EXP_ROOT}/${TAG}/infer"
  local pred_path="${infer_dir}/${pred_name}"
  local stats_path="${infer_dir}/${stats_name}"
  local step2_dir="${EFFR_ROOT}/${effr_group}/step2/${step2_method_dir}"
  local step3_dir="${EFFR_ROOT}/${effr_group}/step3/${step3_method_dir}"
  local step4_dir="${EFFR_ROOT}/${effr_group}/step4"
  local -a extra_detector_args=()
  local -a extra_step2_args=()

  mkdir -p "${infer_dir}" "${step2_dir}" "${step3_dir}" "${step4_dir}"

  if [[ -n "${target_category_id}" ]]; then
    extra_detector_args+=(--target-category-id "${target_category_id}")
    extra_step2_args+=(--target-category-id "${target_category_id}")
  fi
  if [[ -n "${target_category_name}" ]]; then
    extra_detector_args+=(--target-category-name "${target_category_name}")
  fi
  if [[ -n "${class_mapping_json}" ]]; then
    extra_detector_args+=(--class-mapping-json "${class_mapping_json}")
  fi

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
      --force-one-class 0 \
      --v7-exp 0 \
      --v7-confirm-iou 0.4 \
      --v7-confirm-len 2 \
      --v7-confirm-max-age 2 \
      --v7-log-path "${infer_dir}/v7_log.json" \
      --v7-cost-per-frame "${infer_dir}/cost_per_frame.jsonl.gz" \
      --use-confirmation 0 \
      --use-dual-budget 1 \
      --use-targeted-verify 1 \
      --keep-all-categories "${keep_all_categories}" \
      "${extra_detector_args[@]}"
  fi

  "${PYTHON_BIN}" scripts2/build_prevalence_input.py \
    --run-dir "${EXP_ROOT}/${TAG}" \
    --empty-mask "${empty_mask}" \
    --out-dir "${step2_dir}" \
    --category-mode target_only \
    "${extra_step2_args[@]}"

  "${PYTHON_BIN}" scripts2/build_high_conf_fp_labels.py \
    --input "${step2_dir}/prevalence_input.jsonl.gz" \
    --out-dir "${step3_dir}"

  "${PYTHON_BIN}" scripts2/build_step4_prevalence_tables.py \
    --step3-root "${EFFR_ROOT}/${effr_group}/step3" \
    --out-dir "${step4_dir}"
}

case "${TAG}" in
  2026-03-11_v7_noconfirm_caltech_eccv_val)
    run_case \
      "fbdsv" \
      "data_prep/caltech_cct/eval_ready/eccv_val" \
      "eccv_val" \
      "11" \
      "" \
      "1" \
      "" \
      "output/thesis2/analysis/2026-03-11_empty_mask_caltech_eccv_val/empty_mask_strict.jsonl.gz" \
      "caltech_bird" \
      "pred_slim_temporal_sahi_v7_exp0_eccv_val.jsonl.gz" \
      "stats_temporal_sahi_v7_exp0_eccv_val.json" \
      "temporal_sahi_v7_noconfirm" \
      "temporal_sahi_v7_noconfirm"
    ;;
  2026-03-11_v7_noconfirm_coco_bird_absent_val2017)
    run_case \
      "fbdsv" \
      "data_prep/coco_bird_absent/eval_ready/val2017" \
      "val2017" \
      "16" \
      "bird" \
      "0" \
      "configs/coco_bird_only_class_mapping.json" \
      "output/thesis2/analysis/2026-03-11_empty_mask_coco_bird_absent_val2017/empty_mask_strict.jsonl.gz" \
      "coco_bird_absent" \
      "pred_slim_temporal_sahi_v7_exp0_val2017.jsonl.gz" \
      "stats_temporal_sahi_v7_exp0_val2017.json" \
      "temporal_sahi_v7_noconfirm" \
      "temporal_sahi_v7_noconfirm"
    ;;
  2026-03-11_v7_noconfirm_wcs_target05)
    run_case \
      "wcs_subset" \
      "/mnt/ssd960/jm/sahi/data/WCS/subsets/target05" \
      "target05" \
      "1" \
      "" \
      "0" \
      "" \
      "output/thesis2/analysis/2026-03-10_empty_mask_wcs_target05/empty_mask_strict.jsonl.gz" \
      "wcs_target05" \
      "pred_slim_temporal_sahi_v7_exp0_target05.jsonl.gz" \
      "stats_temporal_sahi_v7_exp0_target05.json" \
      "temporal_sahi_v7_noconfirm" \
      "temporal_sahi_v7_noconfirm"
    ;;
  2026-03-11_v7_noconfirm_wcs_target10)
    run_case \
      "wcs_subset" \
      "/mnt/ssd960/jm/sahi/data/WCS/subsets/target10" \
      "target10" \
      "1" \
      "" \
      "0" \
      "" \
      "output/thesis2/analysis/2026-03-10_empty_mask_wcs_target10/empty_mask_strict.jsonl.gz" \
      "wcs_target10" \
      "pred_slim_temporal_sahi_v7_exp0_target10.jsonl.gz" \
      "stats_temporal_sahi_v7_exp0_target10.json" \
      "temporal_sahi_v7_noconfirm" \
      "temporal_sahi_v7_noconfirm"
    ;;
  2026-03-11_v7_noconfirm_wcs_target15)
    run_case \
      "wcs_subset" \
      "/mnt/ssd960/jm/sahi/data/WCS/subsets/target15" \
      "target15" \
      "1" \
      "" \
      "0" \
      "" \
      "output/thesis2/analysis/2026-03-10_empty_mask_wcs_target15/empty_mask_strict.jsonl.gz" \
      "wcs_target15" \
      "pred_slim_temporal_sahi_v7_exp0_target15.jsonl.gz" \
      "stats_temporal_sahi_v7_exp0_target15.json" \
      "temporal_sahi_v7_noconfirm" \
      "temporal_sahi_v7_noconfirm"
    ;;
  2026-03-11_v7_noconfirm_wcs_target20)
    run_case \
      "wcs_subset" \
      "/mnt/ssd960/jm/sahi/data/WCS/subsets/target20" \
      "target20" \
      "1" \
      "" \
      "0" \
      "" \
      "output/thesis2/analysis/2026-03-10_empty_mask_wcs_target20/empty_mask_strict.jsonl.gz" \
      "wcs_target20" \
      "pred_slim_temporal_sahi_v7_exp0_target20.jsonl.gz" \
      "stats_temporal_sahi_v7_exp0_target20.json" \
      "temporal_sahi_v7_noconfirm" \
      "temporal_sahi_v7_noconfirm"
    ;;
  *)
    echo "unknown tag: ${TAG}" >&2
    exit 2
    ;;
esac
