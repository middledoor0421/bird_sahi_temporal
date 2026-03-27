# sahi_temporal/v7/schemas.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DetEvidence:
    has_det: bool
    count: int
    mean_conf: float
    p90_conf: float
    no_det_streak: int


@dataclass
class TrackEvidence:
    num_active: int
    max_miss_streak: int
    instability: float
    recent_lost: bool


@dataclass
class MotionEvidence:
    global_z: float
    local_peak: float


@dataclass
class DiffEvidence:
    p95: float
    p99: float
    std: float
    top1_gap: float


@dataclass
class MetaEvidence:
    prev_mode: Optional[str] = None


@dataclass
class Evidence:
    t: int
    det: DetEvidence
    track: TrackEvidence
    motion: MotionEvidence
    diff: DiffEvidence
    meta: MetaEvidence = field(default_factory=MetaEvidence)

    # Non-JSON-friendly raw data for downstream modules (do not log directly).
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary dict."""
        return {
            "t": int(self.t),
            "det": {
                "has_det": bool(self.det.has_det),
                "count": int(self.det.count),
                "mean_conf": float(self.det.mean_conf),
                "p90_conf": float(self.det.p90_conf),
                "no_det_streak": int(self.det.no_det_streak),
            },
            "track": {
                "num_active": int(self.track.num_active),
                "max_miss_streak": int(self.track.max_miss_streak),
                "instability": float(self.track.instability),
                "recent_lost": bool(self.track.recent_lost),
            },
            "motion": {
                "global_z": float(self.motion.global_z),
                "local_peak": float(self.motion.local_peak),
            },
            "diff": {
                "p95": float(self.diff.p95),
                "p99": float(self.diff.p99),
                "std": float(self.diff.std),
                "top1_gap": float(self.diff.top1_gap),
            },
            "meta": {
                "prev_mode": self.meta.prev_mode,
            },
        }


@dataclass
class ActionPlan:
    mode: str
    risk: float
    K: int
    protect_frac: float
    explore_source: str
    escalate_reason: str
    ttl: Dict[str, int] = field(default_factory=dict)
    cooldown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": str(self.mode),
            "risk": float(self.risk),
            "K": int(self.K),
            "protect_frac": float(self.protect_frac),
            "explore_source": str(self.explore_source),
            "escalate_reason": str(self.escalate_reason),
            "ttl": {str(k): int(v) for k, v in self.ttl.items()},
            "cooldown": {str(k): (v if isinstance(v, (dict, list, str, float, bool)) else int(v)) for k, v in self.cooldown.items()},
        }


@dataclass
class TilePlan:
    tiles: List[int]
    tiles_protect: List[int]
    tiles_explore: List[int]
    tile_mode: str
    K_protect: int
    K_explore: int
    coverage_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Keep tile indices as a short list; logging full coords is unnecessary.
        return {
            "tiles": [int(x) for x in self.tiles],
            "tiles_protect": [int(x) for x in self.tiles_protect],
            "tiles_explore": [int(x) for x in self.tiles_explore],
            "tile_mode": str(self.tile_mode),
            "K_protect": int(self.K_protect),
            "K_explore": int(self.K_explore),
            "coverage_meta": self.coverage_meta,
        }

# Log schema (controller I/O contract)
V7_LOG_SCHEMA_VERSION: str = "v7log_v1"
V7_METHOD_VERSION: str = "v7"

# Required top-level keys for each per-frame log record.
V7_LOG_REQUIRED_KEYS: List[str] = [
    "evidence",
    "plan",
    "tile_plan",
    "pp",
    "confirm",
]

# Required minimal metadata keys.
V7_LOG_REQUIRED_META_KEYS: List[str] = [
    "t",
    "image_id",
    "schema_version",
    "method_version",
]
