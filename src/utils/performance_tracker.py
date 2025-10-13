"""
Performance tracking with generation-by-generation Amdahl and Gustafson law calculations.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class PhaseType(Enum):
    CALIBRATION = "calibration"
    MAIN = "main_algorithm"
    FINAL = "final_evaluation"


@dataclass
class GenerationPerformance:
    """Performance metrics for a single generation"""

    generation: int
    phase: PhaseType
    sequential_time: float
    parallel_time: float
    num_workers: int
    best_fitness: float
    worker_utilization: float
    population_size: int

    # Law calculations
    sequential_fraction: float = 0.0
    parallel_fraction: float = 0.0
    amdahl_speedup: float = 0.0
    gustafson_speedup: float = 0.0
    amdahl_efficiency: float = 0.0
    gustafson_efficiency: float = 0.0

    @property
    def total_time(self) -> float:
        return self.sequential_time + self.parallel_time


@dataclass
class SequentialMeasurement:
    """Individual sequential operation measurement"""

    operation: str
    duration: float
    generation: int
    phase: PhaseType
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """
    Tracks performance and calculates Amdahl/Gustafson laws per generation.
    """

    def __init__(self, gpu_speedup_factor: float = 1.0):
        self.gpu_speedup_factor = gpu_speedup_factor
        self.generation_performances: List[GenerationPerformance] = []
        self.sequential_measurements: List[SequentialMeasurement] = []
        self.current_generation = 0
        self.current_phase: Optional[PhaseType] = None

        # Current generation timing
        self.gen_start_time: Optional[float] = None
        self.sequential_accumulator = 0.0
        self.parallel_start_time: Optional[float] = None

    def start_generation(self, generation: int, phase: PhaseType):
        """Start timing a new generation"""
        self.current_generation = generation
        self.current_phase = phase
        self.gen_start_time = time.perf_counter()
        self.sequential_accumulator = 0.0
        self.parallel_start_time = None

    def record_sequential_time(
        self, duration: float, operation: str = "unknown", metadata: Dict = None
    ):
        """Record time spent in sequential operations"""
        self.sequential_accumulator += duration
        self.sequential_measurements.append(
            SequentialMeasurement(
                operation=operation,
                duration=duration,
                generation=self.current_generation,
                phase=self.current_phase,
                metadata=metadata or {},
            )
        )

    def start_parallel_section(self):
        """Mark the start of parallel evaluation"""
        self.parallel_start_time = time.perf_counter()

    def end_generation(
        self,
        num_workers: int,
        best_fitness: float,
        worker_utilization: float,
        population_size: int,
    ) -> GenerationPerformance:
        """End timing for current generation and calculate performance laws"""
        if self.gen_start_time is None:
            raise ValueError("No generation started")

        total_time = time.perf_counter() - self.gen_start_time

        if self.parallel_start_time:
            parallel_time = time.perf_counter() - self.parallel_start_time
        else:
            parallel_time = 0.0

        sequential_time = self.sequential_accumulator

        (
            sequential_fraction,
            parallel_fraction,
            amdahl_speedup,
            gustafson_speedup,
            amdahl_efficiency,
            gustafson_efficiency,
        ) = self._calculate_performance_laws(
            sequential_time, parallel_time, num_workers
        )

        performance = GenerationPerformance(
            generation=self.current_generation,
            phase=self.current_phase,
            sequential_time=sequential_time,
            parallel_time=parallel_time,
            num_workers=num_workers,
            best_fitness=best_fitness,
            worker_utilization=worker_utilization,
            population_size=population_size,
            sequential_fraction=sequential_fraction,
            parallel_fraction=parallel_fraction,
            amdahl_speedup=amdahl_speedup,
            gustafson_speedup=gustafson_speedup,
            amdahl_efficiency=amdahl_efficiency,
            gustafson_efficiency=gustafson_efficiency,
        )

        self.generation_performances.append(performance)

        self.gen_start_time = None
        self.sequential_accumulator = 0.0
        self.parallel_start_time = None

        return performance

    def _calculate_performance_laws(
        self, sequential_time: float, parallel_time: float, num_workers: int
    ) -> Tuple[float, float, float, float, float, float]:
        """Calculate Amdahl's and Gustafson's laws for given measurements"""
        total_time = sequential_time + parallel_time

        if total_time == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        sequential_fraction = sequential_time / total_time
        parallel_fraction = parallel_time / total_time

        # Adjust for GPU speedup
        effective_workers = num_workers * self.gpu_speedup_factor

        # Amdahl's Law (fixed problem size)
        denominator = sequential_fraction + (
            parallel_fraction / effective_workers
        )
        amdahl_speedup = 1 / denominator if denominator > 0 else float("inf")
        amdahl_efficiency = (
            (amdahl_speedup / effective_workers * 100)
            if effective_workers > 0
            else 0
        )

        # Gustafson's Law (scaled problem size)
        gustafson_speedup = (
            sequential_fraction + parallel_fraction * effective_workers
        )
        gustafson_efficiency = (
            (gustafson_speedup / effective_workers * 100)
            if effective_workers > 0
            else 0
        )

        return (
            sequential_fraction,
            parallel_fraction,
            amdahl_speedup,
            gustafson_speedup,
            amdahl_efficiency,
            gustafson_efficiency,
        )

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
        from src.logger.logger import logger

        perf = self.get_generation_performance(generation)
        if not perf:
            return

        logger.info(f"Sequential Time: {perf.sequential_time:.4f}s")
        logger.info(f"Parallel Time: {perf.parallel_time:.4f}s")
        logger.info(f"Total Time: {perf.total_time:.4f}s")
        logger.info(f"Number of Workers: {perf.num_workers}")
        logger.info(
            f"Effective Workers (GPU-adjusted): {perf.num_workers * self.gpu_speedup_factor:.1f}"
        )

        logger.info("--- SCALABILITY LAWS ---")
        logger.info(
            f"Sequential Fraction: {perf.sequential_fraction:.8f} ({perf.sequential_fraction * 100:.6f}%)"
        )
        logger.info(
            f"Parallel Fraction: {perf.parallel_fraction:.4f} ({perf.parallel_fraction * 100:.1f}%)"
        )
        logger.info(f"Amdahl's Law Speedup: {perf.amdahl_speedup:.2f}x")
        logger.info(f"Gustafson's Law Speedup: {perf.gustafson_speedup:.2f}x")
        logger.info(
            f"Efficiency (Amdahl): {perf.amdahl_efficiency:.2f}% | "
            f"Efficiency (Gustafson): {perf.gustafson_efficiency:.2f}%"
        )

        logger.info("=" * 40)

    def print_overall_report(self):
        """Print comprehensive performance report"""
        from src.logger.logger import logger

        if not self.generation_performances:
            logger.info("No performance data collected")
            return

        total_sequential = sum(
            p.sequential_time for p in self.generation_performances
        )
        total_parallel = sum(
            p.parallel_time for p in self.generation_performances
        )
        total_time = total_sequential + total_parallel

        avg_workers = np.mean(
            [p.num_workers for p in self.generation_performances]
        )

        overall_seq_frac = (
            total_sequential / total_time if total_time > 0 else 0
        )
        overall_par_frac = 1 - overall_seq_frac
        effective_workers = avg_workers * self.gpu_speedup_factor

        denominator = overall_seq_frac + (overall_par_frac / effective_workers)
        overall_amdahl = 1 / denominator if denominator > 0 else float("inf")
        overall_gustafson = (
            overall_seq_frac + overall_par_frac * effective_workers
        )

        logger.info("=" * 40)
        logger.info("OVERALL PERFORMANCE SUMMARY")
        logger.info("=" * 40)
        logger.info(
            f"Total Generations: {len(self.generation_performances) - 1}"
        )
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(
            f"Total Sequential: {total_sequential:.2f}s ({overall_seq_frac * 100:.1f}%)"
        )
        logger.info(
            f"Total Parallel: {total_parallel:.2f}s ({overall_par_frac * 100:.1f}%)"
        )
        logger.info(f"Average Workers: {avg_workers:.1f}")
        logger.info(f"Effective Workers: {effective_workers:.1f}")

        logger.info("\n--- OVERALL SCALABILITY ---")
        logger.info(f"Amdahl's Law Speedup: {overall_amdahl:.2f}x")
        logger.info(f"Gustafson's Law Speedup: {overall_gustafson:.2f}x")

        # Per-phase breakdown
        phases = set(p.phase for p in self.generation_performances)
        for phase in phases:
            phase_perfs = self.get_phase_performances(phase)
            phase_seq = sum(p.sequential_time for p in phase_perfs)
            phase_par = sum(p.parallel_time for p in phase_perfs)
            phase_total = phase_seq + phase_par

            if phase_total > 0:
                phase_seq_frac = phase_seq / phase_total
                logger.info(
                    f"\n{phase.value.upper()} PHASE: {phase_total:.2f}s total"
                )
                logger.info(
                    f"  Sequential: {phase_seq_frac:.3f}, Generations: {len(phase_perfs)}"
                )

        logger.info("=" * 40)


# Global performance tracker
performance_tracker = PerformanceTracker()
