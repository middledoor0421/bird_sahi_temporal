# sahi_temporal/v7/policy.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Dict, Optional

from .schemas import ActionPlan, Evidence


@dataclass
class PolicyConfigV7:
    # Risk thresholds
    mid_risk_thr: float = 0.33
    high_risk_thr: float = 0.66

    # Risk component scales (streaks to normalize)
    det_streak_scale: float = 3.0
    track_miss_scale: float = 3.0

    # Secondary evidence scaling
    motion_z0: float = 0.0
    motion_z_scale: float = 3.0

    # Budget schedule (tiered)
    K_mid_low: int = 2
    K_mid_high: int = 3
    K_high_min: int = 4
    K_high_max: int = 6

    # Full burst safety net (rare)
    full_ttl: int = 2
    csf_high_risk_streak: int = 5
    csf_subset_fail_streak: int = 2

    # Protect fraction defaults
    protect_frac_with_track: float = 0.7
    protect_frac_no_track: float = 0.1

    # Explore source
    explore_source: str = "mixed"

    # Step4: verification burst (triggered by unconfirmed preds)
    verify_burst_len: int = 1
    verify_cooldown: int = 5
    verify_trigger_unconf: int = 1
    verify_only_when_no_confirm: bool = True

    # Empty exploration budget (dataset-calibrated)
    empty_explore_rate: float = 0.10  # tokens per frame
    empty_explore_cap: float = 2.0  # max tokens to allow short bursts
    empty_explore_cost: float = 1.0  # token cost per explore activation

    # Pos-likely exploration budget (separate bucket)
    pos_explore_rate: float = 0.15
    pos_explore_cap: float = 2.0
    pos_explore_cost: float = 1.0

    # Pos-likely signal thresholds (to activate pos budget inside seed-empty segments)
    pos_motion_local_peak_thr: float = 1e9
    pos_diff_ratio_thr: float = 1.5


class PolicyV7:
    """Risk -> Budget -> Strategy policy.

    exp_id is an experiment selector passed from the engine (run_detector --v7-exp).
    """

    def __init__(self, cfg: Optional[PolicyConfigV7] = None, exp_id: int = 0) -> None:
        self.cfg = cfg if cfg is not None else PolicyConfigV7()
        self.exp_id = int(exp_id)

        # Step5: ablation switches (injected by engine when available).
        # These defaults preserve current behavior unless explicitly disabled.
        self.use_dual_budget = True
        self.use_targeted_verify = True

        # Normal policy state
        self._prev_mode: Optional[str] = None
        self._full_ttl_left = 0
        self._high_risk_streak = 0
        self._subset_fail_streak = 0

        # Burst experiment state (exp_id in {60,61,62})
        self._cooldown_full = 0
        self._burst_left = 0
        # Step4: verify burst internal state
        self._verify_left = 0
        self._verify_cooldown_left = 0

        self._empty_tokens = float(self.cfg.empty_explore_cap)
        self._pos_tokens = float(self.cfg.pos_explore_cap)

    def _risk_from_evidence(self, ev: Evidence) -> float:
        det_r = min(1.0, float(ev.det.no_det_streak) / max(1e-6, float(self.cfg.det_streak_scale)))
        track_r = min(1.0, float(ev.track.max_miss_streak) / max(1e-6, float(self.cfg.track_miss_scale)))
        lost_r = 1.0 if bool(ev.track.recent_lost) else 0.0

        mz = float(ev.motion.global_z)
        mz_term = max(0.0, (mz - float(self.cfg.motion_z0)) / max(1e-6, float(self.cfg.motion_z_scale)))
        motion_r = min(1.0, mz_term)

        eps = 1e-6
        diff_proxy = float(ev.diff.top1_gap) / (float(ev.diff.std) + eps)
        diff_r = min(1.0, max(0.0, diff_proxy / 5.0))

        risk = 0.45 * det_r + 0.45 * track_r + 0.25 * lost_r + 0.05 * motion_r + 0.05 * diff_r
        return max(0.0, min(1.0, float(risk)))

    def _is_pos_likely(self, ev: Evidence) -> bool:
        # Pos-likely: allow weak pre-signals even when strong det.has_det is False.
        if bool(ev.det.has_det):
            return True
        try:
            mp = float(getattr(ev.motion, 'local_peak', 0.0))
        except Exception:
            mp = 0.0
        try:
            ratio = float(ev.diff.top1_gap) / max(1e-6, float(ev.diff.std))
        except Exception:
            ratio = 0.0
        if mp >= float(self.cfg.pos_motion_local_peak_thr):
            return True
        if ratio >= float(self.cfg.pos_diff_ratio_thr):
            return True
        return False

    def step(self, ev: Evidence, K_full: int) -> ActionPlan:
        K_full = max(0, int(K_full))

        # If targeted-verify is disabled, clear any pending verify burst state.
        if not bool(getattr(self, "use_targeted_verify", True)):
            self._verify_left = 0
            self._verify_cooldown_left = 0

        # Budget ledger (before)
        empty_before = float(getattr(self, '_empty_tokens', 0.0))
        pos_before = float(getattr(self, '_pos_tokens', 0.0))
        empty_recharge = 0.0
        pos_recharge = 0.0
        empty_spent = 0.0
        pos_spent = 0.0
        bucket_used = "none"
        is_pos_likely = False

        # Token-bucket refill for empty exploration (once per frame)
        self._empty_tokens = min(
            float(self.cfg.empty_explore_cap),
            float(self._empty_tokens) + float(self.cfg.empty_explore_rate),
        )
        if bool(getattr(self, "use_dual_budget", True)):
            self._pos_tokens = min(
                float(self.cfg.pos_explore_cap),
                float(self._pos_tokens) + float(self.cfg.pos_explore_rate),
            )
        else:
            # Ablation: disable positive bucket (pos-likely uses empty bucket only).
            self._pos_tokens = 0.0
        # Budget ledger (recharge)
        empty_recharge = max(0.0, float(getattr(self, '_empty_tokens', 0.0)) - float(empty_before))
        pos_recharge = max(0.0, float(getattr(self, '_pos_tokens', 0.0)) - float(pos_before))
        # Step4: decay verify cooldown every step
        if getattr(self, "_verify_cooldown_left", 0) > 0:
            self._verify_cooldown_left = max(0, int(self._verify_cooldown_left) - 1)

        # Burst experiments
        if int(self.exp_id) in (60, 61, 62):
            return self._step_keyframe_burst(ev=ev, K_full=K_full)

        # Step4: verify burst override (force full_burst for a few frames)
        if getattr(self, "_verify_left", 0) > 0:
            self._verify_left = max(0, int(self._verify_left) - 1)

            mode = "full_burst" if int(K_full) > 0 else "skip"
            K = int(K_full)
            reason = "verify_burst"

            protect_frac = float(self.cfg.protect_frac_with_track) if int(ev.track.num_active) > 0 else float(
                self.cfg.protect_frac_no_track)
            protect_frac = max(0.0, min(1.0, float(protect_frac)))

            plan = ActionPlan(
                mode=str(mode),
                risk=1.0,
                K=int(K),
                protect_frac=float(protect_frac),
                explore_source=str(self.cfg.explore_source),
                escalate_reason=str(reason),
                ttl={
                    "full_ttl_left": int(getattr(self, "_full_ttl_left", 0)),
                    "full_ttl": int(getattr(self.cfg, "full_ttl", 0)),
                    "high_risk_streak": int(getattr(self, "_high_risk_streak", 0)),
                    "subset_fail_streak": int(getattr(self, "_subset_fail_streak", 0)),
                },
                cooldown={
                    "exp_id": int(self.exp_id),
                    "is_full_burst": True,
                    "verify_left": int(self._verify_left),
                    "verify_cooldown_left": int(getattr(self, "_verify_cooldown_left", 0)),
                    "verify_reason": "unconfirmed_trigger",
                },
            )

            self._prev_mode = str(mode)
            return plan

        # ---- Normal v7 policy ----
        K_mid_low = max(0, min(int(self.cfg.K_mid_low), K_full))
        K_mid_high = max(0, min(int(self.cfg.K_mid_high), K_full))
        K_high_min = max(0, min(int(self.cfg.K_high_min), K_full))
        K_high_max = max(0, min(int(self.cfg.K_high_max), K_full))

        risk = self._risk_from_evidence(ev)
        seed_empty = (int(ev.track.num_active) == 0) and (not bool(ev.det.has_det))

        if self._full_ttl_left > 0:
            self._full_ttl_left = max(0, int(self._full_ttl_left) - 1)
            mode = "full" if K_full > 0 else "skip"
            K = int(K_full)
            reason = "ttl"
        else:
            allow_full_burst = (
                    int(self._high_risk_streak) >= int(self.cfg.csf_high_risk_streak)
                    and int(self._subset_fail_streak) >= int(self.cfg.csf_subset_fail_streak)
            )
            if allow_full_burst and K_full > 0:
                mode = "full"
                K = int(K_full)
                reason = "csf_deadline"
                self._full_ttl_left = max(0, int(self.cfg.full_ttl) - 1)
            else:
                if seed_empty:
                    # Default: skip on empty frames. Only explore when risk is high enough.
                    if risk >= float(self.cfg.mid_risk_thr):
                        # Budget gate: allow exploring only if we have enough tokens
                        if bool(getattr(self, "use_dual_budget", True)):
                            is_pos_likely = self._is_pos_likely(ev)
                        else:
                            is_pos_likely = False

                        # Choose bucket
                        if is_pos_likely:
                            tokens = float(self._pos_tokens)
                            cost = float(self.cfg.pos_explore_cost)
                            skip_reason = "pos_budget_skip"
                            explore_reason = "pos_budget_explore"
                        else:
                            tokens = float(self._empty_tokens)
                            cost = float(self.cfg.empty_explore_cost)
                            skip_reason = "empty_budget_skip"
                            explore_reason = "seed_empty_explore"

                        if tokens < cost:
                            mode = "skip"
                            K = 0
                            reason = skip_reason
                        else:
                            # Consume token
                            if is_pos_likely:
                                self._pos_tokens -= cost
                                pos_spent += float(cost)
                                bucket_used = "pos"
                            else:
                                self._empty_tokens -= cost
                                empty_spent += float(cost)
                                bucket_used = "empty"

                            mode = "subset" if K_mid_low > 0 else "skip"
                            alpha = (risk - float(self.cfg.mid_risk_thr)) / max(
                                1e-6, (float(self.cfg.high_risk_thr) - float(self.cfg.mid_risk_thr))
                            )
                            K = int(round(float(K_mid_low) + alpha * float(max(K_mid_high - K_mid_low, 0))))
                            K = max(int(K_mid_low), min(int(K), int(K_mid_high)))
                            reason = explore_reason
                    else:
                        mode = "skip"
                        K = 0
                        reason = "seed_empty_skip"
                elif risk >= float(self.cfg.high_risk_thr):
                    mode = "subset" if K_high_min > 0 else "skip"
                    alpha = (risk - float(self.cfg.high_risk_thr)) / max(1e-6, (1.0 - float(self.cfg.high_risk_thr)))
                    K = int(round(float(K_high_min) + alpha * float(max(K_high_max - K_high_min, 0))))
                    K = max(int(K_high_min), min(int(K), int(K_high_max)))
                    reason = self._pick_reason(ev)
                elif risk >= float(self.cfg.mid_risk_thr):
                    mode = "subset" if K_mid_low > 0 else "skip"
                    alpha = (risk - float(self.cfg.mid_risk_thr)) / max(1e-6, (
                                float(self.cfg.high_risk_thr) - float(self.cfg.mid_risk_thr)))
                    K = int(round(float(K_mid_low) + alpha * float(max(K_mid_high - K_mid_low, 0))))
                    K = max(int(K_mid_low), min(int(K), int(K_mid_high)))
                    reason = self._pick_reason(ev)
                else:
                    mode = "skip"
                    K = 0
                    reason = "low_risk"

        if seed_empty:
            protect_frac = min(0.1, float(self.cfg.protect_frac_no_track))
        elif int(ev.track.num_active) > 0:
            protect_frac = float(self.cfg.protect_frac_with_track)
        else:
            protect_frac = float(self.cfg.protect_frac_no_track)

        protect_frac = max(0.0, min(1.0, float(protect_frac)))

        plan = ActionPlan(
            mode=str(mode),
            risk=float(risk),
            K=int(K),
            protect_frac=float(protect_frac),
            explore_source=str(self.cfg.explore_source),
            escalate_reason=str(reason),
            ttl={
                "full_ttl_left": int(self._full_ttl_left),
                "full_ttl": int(self.cfg.full_ttl),
                "high_risk_streak": int(self._high_risk_streak),
                "subset_fail_streak": int(self._subset_fail_streak),
            },
            cooldown={"exp_id": int(self.exp_id),
                      "empty_tokens": float(getattr(self, "_empty_tokens", 0.0)),
                      "empty_explore_rate": float(self.cfg.empty_explore_rate),
                      "pos_tokens": float(getattr(self, "_pos_tokens", 0.0)),
                      "pos_explore_rate": float(self.cfg.pos_explore_rate),
                      "pos_likely": bool(is_pos_likely),
                      "bucket_used": str(bucket_used),
                      "empty_before": float(empty_before),
                      "empty_after": float(getattr(self, "_empty_tokens", 0.0)),
                      "empty_recharge": float(empty_recharge),
                      "empty_spent": float(empty_spent),
                      "pos_before": float(pos_before),
                      "pos_after": float(getattr(self, "_pos_tokens", 0.0)),
                      "pos_recharge": float(pos_recharge),
                      "pos_spent": float(pos_spent),
                      },
        )

        self._prev_mode = str(mode)
        return plan

    def observe_result(self, plan, ev, has_target_pred: bool,
                       num_unconfirmed: int = 0, num_confirmed: int = 0) -> None:
        risk = self._risk_from_evidence(ev)
        if risk >= float(self.cfg.high_risk_thr):
            self._high_risk_streak += 1
        else:
            self._high_risk_streak = 0

        mode = str(plan.mode).lower().strip()
        seed_empty = (int(ev.track.num_active) == 0) and (not bool(ev.det.has_det))
        focus = seed_empty or (int(ev.det.no_det_streak) > 0)

        if mode in ["subset", "full", "full_burst"] and focus:
            if bool(has_target_pred):
                self._subset_fail_streak = 0
            else:
                self._subset_fail_streak += 1
        else:
            if bool(has_target_pred):
                self._subset_fail_streak = 0

        # Step5 ablation: disable targeted verify burst.
        if not bool(getattr(self, "use_targeted_verify", True)):
            return

        # Step4: trigger verify burst when unconfirmed exists
        nu = int(num_unconfirmed) if num_unconfirmed is not None else 0
        nc = int(num_confirmed) if num_confirmed is not None else 0

        # Cooldown guard
        if self._verify_cooldown_left > 0:
            return

        # Only allow verify on high-risk frames
        # if float(plan.risk) < float(self.cfg.high_risk_thr):
        #     return

        # Trigger condition
        if nu < int(self.cfg.verify_trigger_unconf):
            return

        if bool(self.cfg.verify_only_when_no_confirm) and nc > 0:
            return

        vb = int(self.cfg.verify_burst_len)
        if vb <= 0:
            return

        self._verify_left = vb
        self._verify_cooldown_left = int(self.cfg.verify_cooldown)

    def reset_for_new_video(self) -> None:
        # Step4 verify state reset
        self._verify_left = 0
        self._verify_cooldown_left = 0
        self._empty_tokens = float(self.cfg.empty_explore_cap)
        self._pos_tokens = float(self.cfg.pos_explore_cap)

    @staticmethod
    def _pick_reason(ev: Evidence) -> str:
        if bool(ev.track.recent_lost):
            return "track_lost"
        if int(ev.track.max_miss_streak) > 0:
            return "track_miss"
        if int(ev.det.no_det_streak) > 0:
            return "no_det"
        if float(ev.motion.global_z) > 0.0:
            return "motion"
        return "risk"

    @property
    def prev_mode(self) -> Optional[str]:
        return self._prev_mode

    def _risk_flags_keyframe(self, ev: Evidence) -> Dict[str, bool]:
        seed_empty = (int(ev.track.num_active) == 0) and (not bool(ev.det.has_det))
        hard_case = seed_empty and (int(ev.det.no_det_streak) >= 6)
        low_risk = bool(ev.det.has_det) or (int(ev.track.num_active) > 0 and int(ev.track.max_miss_streak) <= 1)
        mid_risk = (not low_risk) and (not hard_case)
        burst_allowed = int(self._cooldown_full) <= 0
        return {
            "seed_empty": bool(seed_empty),
            "hard_case": bool(hard_case),
            "low_risk": bool(low_risk),
            "mid_risk": bool(mid_risk),
            "burst_allowed": bool(burst_allowed),
        }

    def _burst_params_from_exp(self) -> Dict[str, object]:
        cfg = {
            "K_mid": 3,
            "full_burst_len": 1,
            "full_cooldown": 1,
            "conf_thr_default": 0.25,
            "conf_thr_hard": 0.25,
            "budget_mode": "run1",
        }
        if int(self.exp_id) == 61:
            cfg["conf_thr_hard"] = 0.15
            cfg["budget_mode"] = "run2"
        if int(self.exp_id) == 62:
            cfg["K_mid"] = 4
            cfg["budget_mode"] = "run3"
        return cfg

    def _step_keyframe_burst(self, ev: Evidence, K_full: int) -> ActionPlan:
        if int(self._cooldown_full) > 0:
            self._cooldown_full -= 1

        flags = self._risk_flags_keyframe(ev)
        params = self._burst_params_from_exp()

        seed_empty = bool(flags["seed_empty"])
        hard_case = bool(flags["hard_case"])
        low_risk = bool(flags["low_risk"])
        mid_risk = bool(flags["mid_risk"])
        burst_allowed = bool(flags["burst_allowed"])

        K_mid = int(params["K_mid"])
        full_burst_len = int(params["full_burst_len"])
        full_cooldown = int(params["full_cooldown"])
        conf_thr_default = float(params["conf_thr_default"])
        conf_thr_hard = float(params["conf_thr_hard"])

        mode = "skip"
        K = 0
        is_full_burst = False
        reason = "low_risk"
        conf_thr_used = conf_thr_default

        if low_risk:
            mode = "skip"
            K = 0
            reason = "low_risk"
        elif hard_case and burst_allowed and K_full > 0:
            mode = "full_burst"
            K = int(K_full)
            is_full_burst = True
            reason = "hard_case_full"
            self._cooldown_full = int(full_cooldown)
            self._burst_left = int(full_burst_len) - 1
            conf_thr_used = conf_thr_hard
        elif hard_case:
            mode = "subset"
            K = int(K_mid)
            reason = "hard_case_subset"
            conf_thr_used = conf_thr_hard
        elif mid_risk:
            mode = "subset"
            K = int(K_mid)
            reason = "mid_risk"
        else:
            mode = "subset" if seed_empty else "skip"
            K = int(K_mid) if seed_empty else 0
            reason = "fallback"

        if int(self._burst_left) > 0 and K_full > 0:
            mode = "full_burst"
            K = int(K_full)
            is_full_burst = True
            reason = "hard_case_full_cont"
            self._burst_left -= 1
            conf_thr_used = conf_thr_hard

        plan = ActionPlan(
            mode=str(mode),
            risk=0.0,
            K=int(K),
            protect_frac=0.0,
            explore_source=str(self.cfg.explore_source),
            escalate_reason=str(reason),
            ttl={},
            cooldown={
                "exp_id": int(self.exp_id),
                "cooldown_full": int(self._cooldown_full),
                "is_full_burst": bool(is_full_burst),
                "conf_thr_used": float(conf_thr_used),
                "K_mid": int(K_mid),
                "budget_mode": str(params.get("budget_mode", "")),
            },
        )

        self._prev_mode = str(mode)
        return plan
