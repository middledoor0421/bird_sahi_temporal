# sahi_temporal/v6/policy.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Dict, Optional

from .schemas import ActionPlan, Evidence


@dataclass
class PolicyConfigV6:
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


class PolicyV6:
    """Risk -> Budget -> Strategy policy.

    exp_id is an experiment selector passed from the engine (run_detector --v6-exp).
    """

    def __init__(self, cfg: Optional[PolicyConfigV6] = None, exp_id: int = 0) -> None:
        self.cfg = cfg if cfg is not None else PolicyConfigV6()
        self.exp_id = int(exp_id)

        # Normal policy state
        self._prev_mode: Optional[str] = None
        self._full_ttl_left = 0
        self._high_risk_streak = 0
        self._subset_fail_streak = 0

        # Burst experiment state (exp_id in {60,61,62})
        self._cooldown_full = 0
        self._burst_left = 0

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

    def step(self, ev: Evidence, K_full: int) -> ActionPlan:
        K_full = max(0, int(K_full))

        # Burst experiments
        if int(self.exp_id) in (60, 61, 62):
            return self._step_keyframe_burst(ev=ev, K_full=K_full)

        # ---- Normal v6 policy ----
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
                    mode = "subset" if K_high_min > 0 else "skip"
                    K = int(max(K_high_min, K_mid_high))
                    reason = "seed_empty"
                elif risk >= float(self.cfg.high_risk_thr):
                    mode = "subset" if K_high_min > 0 else "skip"
                    alpha = (risk - float(self.cfg.high_risk_thr)) / max(1e-6, (1.0 - float(self.cfg.high_risk_thr)))
                    K = int(round(float(K_high_min) + alpha * float(max(K_high_max - K_high_min, 0))))
                    K = max(int(K_high_min), min(int(K), int(K_high_max)))
                    reason = self._pick_reason(ev)
                elif risk >= float(self.cfg.mid_risk_thr):
                    mode = "subset" if K_mid_low > 0 else "skip"
                    alpha = (risk - float(self.cfg.mid_risk_thr)) / max(1e-6, (float(self.cfg.high_risk_thr) - float(self.cfg.mid_risk_thr)))
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
            cooldown={"exp_id": int(self.exp_id)},
        )

        self._prev_mode = str(mode)
        return plan

    def observe_result(self, plan: ActionPlan, ev: Evidence, has_target_pred: bool) -> None:
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
