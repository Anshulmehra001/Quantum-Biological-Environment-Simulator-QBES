"""
Performance profiling and optimization tools for QBES.

This module provides comprehensive performance analysis including:
- Timing measurements
- Memory profiling
- Bottleneck identification
- Optimization recommendations
"""

import time
import logging
import psutil
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import tracemalloc


@dataclass
class PerformanceMetrics:
    """Performance metrics for a simulation or operation."""
    operation: str
    wall_time: float  # seconds
    cpu_time: float  # seconds
    memory_peak: float  # MB
    memory_increase: float  # MB
    calls: int = 1
    timestamps: List[float] = field(default_factory=list)
    
    def __str__(self) -> str:
        """Format metrics as string."""
        return (f"{self.operation}: {self.wall_time:.3f}s wall, "
                f"{self.cpu_time:.3f}s CPU, "
                f"{self.memory_peak:.1f}MB peak, "
                f"{self.memory_increase:.1f}MB increase")


@dataclass
class ProfilingSession:
    """Container for profiling session results."""
    session_name: str
    start_time: float
    end_time: float
    metrics: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    
    @property
    def total_time(self) -> float:
        """Total session time."""
        return self.end_time - self.start_time
    
    def add_metric(self, metric: PerformanceMetrics):
        """Add performance metric to session."""
        self.metrics[metric.operation] = metric
    
    def get_bottlenecks(self, threshold: float = 0.1) -> List[str]:
        """
        Identify bottlenecks (operations taking >threshold of total time).
        
        Args:
            threshold: Fraction of total time (0-1)
            
        Returns:
            List of operation names that are bottlenecks
        """
        bottlenecks = []
        threshold_time = self.total_time * threshold
        
        for op, metric in self.metrics.items():
            if metric.wall_time > threshold_time:
                bottlenecks.append(op)
        
        return bottlenecks
    
    def generate_report(self) -> str:
        """Generate formatted performance report."""
        lines = []
        lines.append("="*80)
        lines.append(f"PERFORMANCE PROFILING REPORT: {self.session_name}")
        lines.append("="*80)
        lines.append(f"Total Time: {self.total_time:.3f}s")
        lines.append("")
        
        # Sort by wall time descending
        sorted_metrics = sorted(self.metrics.items(), 
                              key=lambda x: x[1].wall_time, 
                              reverse=True)
        
        lines.append("Time Breakdown:")
        lines.append("-"*80)
        lines.append(f"{'Operation':<40} {'Time (s)':<12} {'CPU %':<10} {'Memory (MB)':<15}")
        lines.append("-"*80)
        
        for op, metric in sorted_metrics:
            cpu_percent = (metric.cpu_time / metric.wall_time * 100) if metric.wall_time > 0 else 0
            lines.append(f"{op:<40} {metric.wall_time:<12.3f} {cpu_percent:<10.1f} {metric.memory_peak:<15.1f}")
        
        lines.append("")
        
        # Bottlenecks
        bottlenecks = self.get_bottlenecks(0.1)
        if bottlenecks:
            lines.append("⚠️  Performance Bottlenecks (>10% of total time):")
            for bn in bottlenecks:
                percent = (self.metrics[bn].wall_time / self.total_time) * 100
                lines.append(f"  - {bn}: {percent:.1f}% of total time")
            lines.append("")
        
        # Memory analysis
        max_mem = max(m.memory_peak for m in self.metrics.values()) if self.metrics else 0
        total_increase = sum(m.memory_increase for m in self.metrics.values())
        
        lines.append("Memory Usage:")
        lines.append(f"  Peak: {max_mem:.1f} MB")
        lines.append(f"  Total Increase: {total_increase:.1f} MB")
        lines.append("")
        
        lines.append("="*80)
        return "\n".join(lines)


class PerformanceProfiler:
    """
    Performance profiler for QBES operations.
    
    Tracks timing, memory usage, and identifies bottlenecks.
    """
    
    def __init__(self, name: str = "QBES Profiling"):
        """
        Initialize profiler.
        
        Args:
            name: Name for this profiling session
        """
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.sessions: List[ProfilingSession] = []
        self.current_session: Optional[ProfilingSession] = None
        self._operation_stack: List[tuple] = []
    
    def start_session(self, session_name: str):
        """Start a new profiling session."""
        self.current_session = ProfilingSession(
            session_name=session_name,
            start_time=time.perf_counter(),
            end_time=0.0
        )
        self.logger.info(f"Started profiling session: {session_name}")
    
    def end_session(self):
        """End current profiling session."""
        if self.current_session:
            self.current_session.end_time = time.perf_counter()
            self.sessions.append(self.current_session)
            self.logger.info(f"Ended profiling session: {self.current_session.session_name}")
            report = self.current_session.generate_report()
            self.logger.info(f"\n{report}")
            self.current_session = None
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """
        Context manager for profiling an operation.
        
        Args:
            operation_name: Name of the operation being profiled
            
        Example:
            >>> profiler = PerformanceProfiler()
            >>> profiler.start_session("Simulation")
            >>> with profiler.profile_operation("MD Integration"):
            ...     run_md_simulation()
        """
        # Start tracking
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Track memory allocations
        tracemalloc.start()
        
        try:
            yield
        finally:
            # End tracking
            end_wall = time.perf_counter()
            end_cpu = time.process_time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / 1024 / 1024
            
            # Create metrics
            metric = PerformanceMetrics(
                operation=operation_name,
                wall_time=end_wall - start_wall,
                cpu_time=end_cpu - start_cpu,
                memory_peak=peak_mb,
                memory_increase=end_memory - start_memory
            )
            
            # Add to current session
            if self.current_session:
                self.current_session.add_metric(metric)
            else:
                self.logger.warning(f"No active session for operation: {operation_name}")
            
            self.logger.debug(str(metric))
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator for profiling a function.
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function that tracks performance
            
        Example:
            >>> profiler = PerformanceProfiler()
            >>> @profiler.profile_function
            ... def expensive_operation():
            ...     time.sleep(1)
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile_operation(func.__name__):
                return func(*args, **kwargs)
        return wrapper
    
    def get_latest_report(self) -> str:
        """Get report from most recent session."""
        if not self.sessions:
            return "No profiling sessions recorded."
        return self.sessions[-1].generate_report()
    
    def get_bottlenecks(self, session_index: int = -1) -> List[str]:
        """Get bottlenecks from specific session."""
        if not self.sessions:
            return []
        return self.sessions[session_index].get_bottlenecks()
    
    def compare_sessions(self, 
                        session1_idx: int, 
                        session2_idx: int) -> Dict[str, Dict]:
        """
        Compare two profiling sessions.
        
        Args:
            session1_idx: Index of first session
            session2_idx: Index of second session
            
        Returns:
            Dictionary comparing metrics between sessions
        """
        if len(self.sessions) <= max(session1_idx, session2_idx):
            return {}
        
        s1 = self.sessions[session1_idx]
        s2 = self.sessions[session2_idx]
        
        comparison = {}
        
        # Compare common operations
        common_ops = set(s1.metrics.keys()) & set(s2.metrics.keys())
        
        for op in common_ops:
            m1 = s1.metrics[op]
            m2 = s2.metrics[op]
            
            speedup = m1.wall_time / m2.wall_time if m2.wall_time > 0 else float('inf')
            memory_change = m2.memory_peak - m1.memory_peak
            
            comparison[op] = {
                'time_change': m2.wall_time - m1.wall_time,
                'speedup': speedup,
                'memory_change': memory_change,
                'improved': m2.wall_time < m1.wall_time
            }
        
        return comparison


class OptimizationAnalyzer:
    """
    Analyze performance and provide optimization recommendations.
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_simulation_performance(self, 
                                      session: ProfilingSession) -> List[str]:
        """
        Analyze simulation performance and suggest optimizations.
        
        Args:
            session: Profiling session to analyze
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Check for time bottlenecks
        bottlenecks = session.get_bottlenecks(threshold=0.15)
        
        for bn in bottlenecks:
            metric = session.metrics[bn]
            
            if 'md' in bn.lower() or 'molecular' in bn.lower():
                recommendations.append(
                    f"⚡ MD Integration ({bn}) takes {metric.wall_time:.1f}s. "
                    "Consider: smaller timestep, fewer atoms, or GPU acceleration"
                )
            
            if 'quantum' in bn.lower() or 'lindblad' in bn.lower():
                recommendations.append(
                    f"⚡ Quantum simulation ({bn}) takes {metric.wall_time:.1f}s. "
                    "Consider: adaptive timestep, sparse matrices, or lower basis size"
                )
            
            if 'hamiltonian' in bn.lower():
                recommendations.append(
                    f"⚡ Hamiltonian construction ({bn}) takes {metric.wall_time:.1f}s. "
                    "Consider: caching, symmetry exploitation, or precomputation"
                )
        
        # Check memory usage
        max_memory = max(m.memory_peak for m in session.metrics.values())
        
        if max_memory > 1000:  # > 1 GB
            recommendations.append(
                f"💾 High memory usage ({max_memory:.0f}MB). "
                "Consider: chunked processing, data streaming, or memory profiling"
            )
        
        # Check CPU efficiency
        for op, metric in session.metrics.items():
            cpu_efficiency = (metric.cpu_time / metric.wall_time) if metric.wall_time > 0 else 0
            
            if cpu_efficiency < 0.5 and metric.wall_time > 1.0:
                recommendations.append(
                    f"🔄 Low CPU utilization in {op} ({cpu_efficiency*100:.0f}%). "
                    "Consider: parallelization or reducing I/O overhead"
                )
        
        if not recommendations:
            recommendations.append("✅ Performance looks good! No major bottlenecks detected.")
        
        return recommendations
    
    def estimate_scaling(self, 
                        metrics: PerformanceMetrics, 
                        current_size: int,
                        target_size: int) -> Dict[str, float]:
        """
        Estimate performance scaling to larger system sizes.
        
        Args:
            metrics: Current performance metrics
            current_size: Current system size (atoms/sites)
            target_size: Target system size
            
        Returns:
            Estimated metrics for target size
        """
        ratio = target_size / current_size
        
        # Different scaling for different operations
        # MD: O(N) to O(N²) depending on cutoffs
        # Quantum: O(N²) to O(N³) for matrix operations
        
        # Conservative estimate: O(N²) scaling
        time_scaling = ratio ** 2
        memory_scaling = ratio ** 1.5  # Slightly sublinear
        
        estimates = {
            'estimated_time': metrics.wall_time * time_scaling,
            'estimated_memory': metrics.memory_peak * memory_scaling,
            'scaling_factor': ratio,
            'feasible': metrics.memory_peak * memory_scaling < 8000  # < 8GB
        }
        
        return estimates


# Convenience functions
def profile_simulation(func: Callable) -> Callable:
    """
    Decorator to automatically profile a simulation function.
    
    Example:
        >>> @profile_simulation
        ... def run_my_simulation(config):
        ...     # simulation code
        ...     pass
    """
    profiler = PerformanceProfiler()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler.start_session(f"Simulation: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.end_session()
    
    return wrapper


def quick_profile(operation_name: str = "Operation"):
    """
    Quick profiling context manager for ad-hoc performance testing.
    
    Example:
        >>> with quick_profile("Data Processing"):
        ...     process_large_dataset()
    """
    return PerformanceProfiler().profile_operation(operation_name)
