from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List

import numpy as np


@dataclass
class MaxStepsCurriculumConfig:
    enable_max_steps_control: bool = False
    update_mode: str = 'acl'

    min_max_steps: int = 1
    max_max_steps: int = 64
    headroom_steps: int = 2
    smoothing_alpha: float = 0.2

    # non-core but stability / observability
    success_sample_window: int = 100
    metrics_window: int = 50
    min_success_samples_for_update_turns: int = 10

    # if outcome is float, treat success as outcome >= threshold
    success_outcome_threshold: float = 0.5
    total_steps: int = 200



class MaxStepsCurriculumController:
    """ACL-V3-style controller (lightweight) for multi-turn max_steps.

    Core idea:
      - maintain a pool of *successful* episode turns
      - estimate capability boundary as P90(success_turns)
      - target = P90 + headroom
      - current <- EMA(current, target) with smoothing_alpha

    This controller ONLY manages max_steps (no response length / token budget control).
    """

    def __init__(
        self,
        cfg: MaxStepsCurriculumConfig,
        initial_current_max_steps: int,
    ):
        self.cfg = cfg
        self.current_max_steps: float = float(initial_current_max_steps)
        self.initial_max_steps: float = float(initial_current_max_steps)

        self._success_turns: Deque[int] = deque(maxlen=int(cfg.success_sample_window))
        self._success_rate_window: Deque[float] = deque(maxlen=int(cfg.metrics_window))
        self._clip_ratio_window: Deque[float] = deque(maxlen=int(cfg.metrics_window))

        self._last_logged_int: int = int(round(self.current_max_steps))


    def get_max_steps_override(self) -> int:
        applied = int(round(self.current_max_steps))
        applied = int(np.clip(applied, self.cfg.min_max_steps, self.cfg.max_max_steps))
        return applied

    def success_samples_count(self) -> int:
        return len(self._success_turns)


    def update(
        self,
        *,
        success_rate: float,
        clip_ratio: float,
        successful_turns: List[int],
        epoch: int=0, 
        step: int=0,
    ) -> Dict[str, float | int | str]:
        """Update controller state and return a metrics dict.

        Returns keys (subset):
          - curriculum/current_max_steps
          - curriculum/success_samples_count
          - curriculum/avg_success_rate
          - curriculum/avg_exceed_rate
          - debug/p90_success_turns, debug/target_max_steps
          - debug/reason
        """
        if self.cfg.update_mode == 'tti':
            return self.update_tti(
                success_rate=success_rate,
                clip_ratio=clip_ratio,
                successful_turns=successful_turns,
                epoch=epoch,
            )
        if self.cfg.update_mode == 'scaling':
            return self.update_scaling(
                success_rate=success_rate,
                clip_ratio=clip_ratio,
                successful_turns=successful_turns,
                current_step=step,
            )

        self._success_rate_window.append(float(success_rate))
        self._clip_ratio_window.append(float(clip_ratio))

        for t in successful_turns:
            if t is None:
                continue
            try:
                t_int = int(t)
            except Exception:
                continue
            if t_int > 0:
                self._success_turns.append(t_int)

        old = float(self.current_max_steps)
        reason = "insufficient_success_samples"
        p90 = None
        target = None

        if (not self.cfg.enable_max_steps_control) or (len(self._success_turns) < int(self.cfg.min_success_samples_for_update_turns)):
            reason = "disabled" if not self.cfg.enable_max_steps_control else "insufficient_success_samples"
        else:
            p90 = float(np.percentile(np.array(list(self._success_turns), dtype=np.float32), 90))
            target = p90 + float(self.cfg.headroom_steps)
            # smooth
            a = float(self.cfg.smoothing_alpha)
            self.current_max_steps = (1.0 - a) * self.current_max_steps + a * target
            # clip
            self.current_max_steps = float(np.clip(self.current_max_steps, self.cfg.min_max_steps, self.cfg.max_max_steps))
            reason = "P90 + headroom"

        new = float(self.current_max_steps)

        avg_success = float(np.mean(self._success_rate_window)) if self._success_rate_window else 0.0
        avg_exceed = float(np.mean(self._clip_ratio_window)) if self._clip_ratio_window else 0.0

        metrics: Dict[str, float | int | str] = {
            "curriculum/current_max_steps": new,
            "curriculum/success_samples_count": len(self._success_turns),
            "curriculum/avg_success_rate": avg_success,
            "curriculum/avg_exceed_rate": avg_exceed,  # defined as rolling mean clip_ratio
            "curriculum/debug/old_current_max_steps": old,
            "curriculum/debug/reason": reason,
        }
        if p90 is not None:
            metrics["curriculum/debug/p90_success_turns"] = p90
        if target is not None:
            metrics["curriculum/debug/target_max_steps"] = float(target)

        return metrics

    def should_log_step_change(self) -> bool:
        cur_int = int(round(self.current_max_steps))
        if abs(cur_int - self._last_logged_int) >= 1:
            self._last_logged_int = cur_int
            return True
        return False

    def update_tti(
        self,
        *,
        success_rate: float,
        clip_ratio: float,
        successful_turns: List[int],
        epoch: int,
    ) -> Dict[str, float | int | str]:
        """Update controller state using TTI linear schedule logic.
        Logic: current_max_steps = min(initial_max_steps * epoch, max_max_steps)
        """
        # Maintain metrics history for observability
        self._success_rate_window.append(float(success_rate))
        self._clip_ratio_window.append(float(clip_ratio))

        for t in successful_turns:
            if t is None:
                continue
            try:
                t_int = int(t)
            except Exception:
                continue
            if t_int > 0:
                self._success_turns.append(t_int)

        old = float(self.current_max_steps)
        
        # Linear schedule logic based on epoch
        target = float(self.initial_max_steps * (epoch+1))
        self.current_max_steps = min(target, float(self.cfg.max_max_steps))
        
        reason = "tti_linear_schedule"
        new = float(self.current_max_steps)

        avg_success = float(np.mean(self._success_rate_window)) if self._success_rate_window else 0.0
        avg_exceed = float(np.mean(self._clip_ratio_window)) if self._clip_ratio_window else 0.0

        metrics: Dict[str, float | int | str] = {
            "curriculum/current_max_steps": new,
            "curriculum/success_samples_count": len(self._success_turns),
            "curriculum/avg_success_rate": avg_success,
            "curriculum/avg_exceed_rate": avg_exceed,
            "curriculum/debug/old_current_max_steps": old,
            "curriculum/debug/reason": reason,
            "curriculum/debug/target_max_steps": target,
        }

        return metrics

    def update_scaling(
        self,
        *,
        success_rate: float,
        clip_ratio: float,
        successful_turns: List[int],
        current_step: int,
    ) -> Dict[str, float | int | str]:
        self._success_rate_window.append(float(success_rate))
        self._clip_ratio_window.append(float(clip_ratio))

        for t in successful_turns:
            if t is None:
                continue
            try:
                t_int = int(t)
            except Exception:
                continue
            if t_int > 0:
                self._success_turns.append(t_int)

        old = float(self.current_max_steps)
        target = old + (current_step / self.cfg.total_steps) * (self.cfg.max_max_steps - self.initial_max_steps)
        self.current_max_steps = min(target, float(self.cfg.max_max_steps))
        reason = "scaling"
        new = float(self.current_max_steps)
        avg_success = float(np.mean(self._success_rate_window)) if self._success_rate_window else 0.0
        avg_exceed = float(np.mean(self._clip_ratio_window)) if self._clip_ratio_window else 0.0

        metrics: Dict[str, float | int | str] = {
            "curriculum/current_max_steps": new,
            "curriculum/success_samples_count": len(self._success_turns),
            "curriculum/avg_success_rate": avg_success,
            "curriculum/avg_exceed_rate": avg_exceed,
            "curriculum/debug/old_current_max_steps": old,
            "curriculum/debug/reason": reason,
            "curriculum/debug/target_max_steps": target,
        }

        return metrics
