"""
Performance tracking with generation-by-generation Amdahl and Gustafson law calculations.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from src.logger.logger import logger


class PhaseType(Enum):
    CALIBRATION = "calibration"
    MAIN = "main_algorithm"
    FINAL = "final_evaluation"


@dataclass
class GenerationPerformance:
    generation: int
    phase: PhaseType
    sequential_time: float
    parallel_time: float
    untracked_time: float
    wall_time: float
    clock_time: float
    num_workers: int
    best_fitness: float
    worker_utilization: float
    population_size: int


@dataclass
class SequentialMeasurement:
    """Individual sequential operation measurement"""

    operation: str
    duration: float
    generation: int
    phase: Optional[PhaseType]


class _PerformanceTracker:
    """
    Tracks performance and calculates Amdahl/Gustafson laws per generation.
    Measures both wall time (time.perf_counter) and clock time (time.process_time).
    """

    def __init__(self, gpu_speedup_factor: float = 1.0):
        self.parallel_start_wall_time = None
        self.gen_start_clock_time = None
        self.gen_start_wall_time = None
        self.current_phase = None
        self.current_generation = None

        self.gpu_speedup_factor = gpu_speedup_factor
        self.generation_performances: List[GenerationPerformance] = []
        self.sequential_measurements: List[SequentialMeasurement] = []
        self._reset_generation_state()

    def _reset_generation_state(self):
        """Helper to reset all per-generation tracking variables."""
        self.current_generation: Optional[int] = None
        self.current_phase: Optional[PhaseType] = None
        self.gen_start_wall_time: Optional[float] = None
        self.gen_start_clock_time: Optional[float] = None
        self.sequential_accumulator: float = 0.0
        self.parallel_start_wall_time: Optional[float] = None

    def start_generation(self, generation: int, phase: PhaseType):
        """Start timing a new generation"""
        if self.gen_start_wall_time is not None:
            raise RuntimeError(
                f"Cannot start generation {generation} because generation {self.current_generation} is still running."
            )
        self._reset_generation_state()
        self.current_generation = generation
        self.current_phase = phase
        self.gen_start_wall_time = time.perf_counter()
        self.gen_start_clock_time = time.process_time()

    def record_sequential_time(
        self,
        duration: float,
        operation: str = "unknown",
    ):
        """Record time spent in sequential operations"""
        if self.gen_start_wall_time is None:
            raise RuntimeError(
                "Cannot record sequential time before starting a generation."
            )
        self.sequential_accumulator += duration
        self.sequential_measurements.append(
            SequentialMeasurement(
                operation=operation,
                duration=duration,
                generation=self.current_generation,
                phase=self.current_phase,
            )
        )

    def start_parallel_section(self):
        """Mark the start of parallel evaluation"""
        if self.gen_start_wall_time is None:
            raise RuntimeError(
                "Cannot start parallel section before starting a generation."
            )
        self.parallel_start_wall_time = time.perf_counter()

    def end_generation(
        self,
        num_workers: int,
        best_fitness: float,
        worker_utilization: float,
        population_size: int,
    ) -> GenerationPerformance:
        """End timing for current generation and calculate performance laws"""
        if (
            self.gen_start_wall_time is None
            or self.gen_start_clock_time is None
        ):
            raise RuntimeError(
                "Cannot end generation because no generation was started."
            )

        end_wall_time = time.perf_counter()
        end_clock_time = time.process_time()

        total_wall_time = end_wall_time - self.gen_start_wall_time
        total_clock_time = end_clock_time - self.gen_start_clock_time

        parallel_time = (
            (end_wall_time - self.parallel_start_wall_time)
            if self.parallel_start_wall_time
            else 0.0
        )

        sequential_time = self.sequential_accumulator
        untracked_time = total_wall_time - sequential_time - parallel_time

        performance = GenerationPerformance(
            generation=self.current_generation,
            phase=self.current_phase,
            sequential_time=sequential_time,
            parallel_time=parallel_time,
            untracked_time=untracked_time,
            wall_time=total_wall_time,
            clock_time=total_clock_time,
            num_workers=num_workers,
            best_fitness=best_fitness,
            worker_utilization=worker_utilization,
            population_size=population_size,
        )

        self.generation_performances.append(performance)

        # Reset state for the next generation
        self._reset_generation_state()

        return performance

    def get_generation_performance(
        self, generation: int
    ) -> Optional[GenerationPerformance]:
        """Get performance data for specific generation"""
        for perf in self.generation_performances:
            if perf.generation == generation:
                return perf
        return None

    def get_phase_performances(
        self, phase: PhaseType
    ) -> List[GenerationPerformance]:
        """Get all performances for a specific phase"""
        return [p for p in self.generation_performances if p.phase == phase]

    def print_generation_report(self, generation: int):
        """Print performance report for a specific generation"""
        perf = self.get_generation_performance(generation)
        if not perf:
            logger.warning(f"No performance data for generation {generation}")
            return

        logger.info(
            f"--- PERFORMANCE REPORT FOR GENERATION {perf.generation} (Phase: {perf.phase.value}) ---"
        )
        logger.info(
            f"Wall Time: {perf.wall_time:.8f}s | Clock Time: {perf.clock_time:.8f}s"
        )
        logger.info(f"Sequential Time: {perf.sequential_time:.8f}s")
        logger.info(f"Parallel Time: {perf.parallel_time:.8f}s")
        logger.info(f"Untracked Time: {perf.untracked_time:.8f}s")
        logger.info(f"Number of Workers: {perf.num_workers}")
        logger.info(
            f"Effective Workers (GPU-adjusted): {perf.num_workers * self.gpu_speedup_factor:.1f}"
        )

        logger.info("=" * 60)

    def print_overall_report(self):
        """Print comprehensive performance report"""
        if not self.generation_performances:
            logger.info("No performance data collected")
            return

        total_seq = sum(p.sequential_time for p in self.generation_performances)
        total_par = sum(p.parallel_time for p in self.generation_performances)
        total_wall = sum(p.wall_time for p in self.generation_performances)
        total_clock = sum(p.clock_time for p in self.generation_performances)

        avg_workers = np.mean(
            [p.num_workers for p in self.generation_performances]
        )

        # Calculate overall fractions based on the sum of times
        overall_seq_frac = total_seq / total_wall if total_wall > 0 else 0
        overall_par_frac = total_par / total_wall if total_wall > 0 else 0

        effective_workers = avg_workers * self.gpu_speedup_factor

        logger.info("=" * 40)
        logger.info("OVERALL PERFORMANCE SUMMARY")
        logger.info("=" * 40)
        logger.info(f"Total Generations: {len(self.generation_performances)}")
        logger.info(f"Total Wall Time: {total_wall:.2f}s")
        logger.info(f"Total Clock Time: {total_clock:.2f}s")
        logger.info(
            f"Total Sequential: {total_seq:.2f}s ({overall_seq_frac * 100:.1f}%)"
        )
        logger.info(
            f"Total Parallel: {total_par:.2f}s ({overall_par_frac * 100:.1f}%)"
        )
        logger.info(f"Average Workers: {avg_workers:.1f}")
        logger.info(f"Effective Workers: {effective_workers:.1f}")

        # Per-phase breakdown
        phases = sorted(
            list(set(p.phase for p in self.generation_performances)),
            key=lambda x: x.value,
        )
        for phase in phases:
            phase_perfs = self.get_phase_performances(phase)
            phase_seq = sum(p.sequential_time for p in phase_perfs)
            phase_wall = sum(p.wall_time for p in phase_perfs)

            if phase_wall > 0:
                phase_seq_frac = phase_seq / phase_wall
                logger.info(
                    f"\n{phase.value.upper()} PHASE: {phase_wall:.2f}s total wall time"
                )
                logger.info(
                    f"  Sequential Fraction: {phase_seq_frac:.3f}, Generations: {len(phase_perfs)}"
                )

        logger.info("=" * 40)


# Global performance tracker
performance_tracker = _PerformanceTracker()
