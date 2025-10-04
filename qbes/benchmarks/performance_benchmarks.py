"""
Performance benchmarking for computational scaling analysis.

This module provides tools to analyze how QBES performance scales
with system size and complexity.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from .benchmark_systems import BenchmarkRunner, HarmonicOscillatorBenchmark
from ..core.data_models import SimulationConfig


@dataclass
class PerformanceResult:
    """Results from performance benchmarking."""
    system_size: int
    computation_time: float
    memory_usage: Optional[float] = None
    convergence_achieved: bool = True
    error_message: Optional[str] = None


class PerformanceBenchmarker:
    """
    Performance benchmarking suite for QBES.
    
    Tests computational scaling with system size and provides
    analysis of performance bottlenecks.
    """
    
    def __init__(self):
        """Initialize performance benchmarker."""
        self.results: List[PerformanceResult] = []
    
    def test_system_size_scaling(self, 
                                system_sizes: List[int] = None,
                                simulation_time: float = 0.1,
                                time_step: float = 0.01) -> List[PerformanceResult]:
        """
        Test how computation time scales with quantum system size.
        
        Args:
            system_sizes: List of system sizes (number of levels) to test
            simulation_time: Duration of each benchmark simulation
            time_step: Time step for evolution
            
        Returns:
            List of performance results
        """
        if system_sizes is None:
            system_sizes = [2, 4, 8, 16, 32]
        
        self.results = []
        
        print("Testing computational scaling with system size...")
        print("=" * 50)
        
        for size in system_sizes:
            print(f"Testing system size: {size}")
            
            try:
                # Create harmonic oscillator benchmark with specified size
                benchmark = HarmonicOscillatorBenchmark(n_levels=size)
                
                # Measure computation time
                start_time = time.time()
                result = benchmark.run_benchmark(simulation_time, time_step)
                computation_time = time.time() - start_time
                
                # Check if benchmark passed (convergence achieved)
                convergence_achieved = result.test_passed
                
                perf_result = PerformanceResult(
                    system_size=size,
                    computation_time=computation_time,
                    convergence_achieved=convergence_achieved
                )
                
                self.results.append(perf_result)
                
                print(f"  Time: {computation_time:.3f}s")
                print(f"  Convergence: {'Yes' if convergence_achieved else 'No'}")
                
                if not convergence_achieved:
                    print(f"  Error: {result.error_message}")
                
            except Exception as e:
                error_result = PerformanceResult(
                    system_size=size,
                    computation_time=0.0,
                    convergence_achieved=False,
                    error_message=str(e)
                )
                self.results.append(error_result)
                print(f"  FAILED: {str(e)}")
            
            print()
        
        return self.results
    
    def analyze_scaling_behavior(self) -> Dict[str, float]:
        """
        Analyze computational scaling behavior.
        
        Fits scaling laws to the performance data and returns
        scaling exponents.
        
        Returns:
            Dictionary with scaling analysis results
        """
        if not self.results:
            raise ValueError("No performance results available. Run benchmarks first.")
        
        # Extract successful results
        successful_results = [r for r in self.results if r.convergence_achieved]
        
        if len(successful_results) < 3:
            raise ValueError("Need at least 3 successful results for scaling analysis.")
        
        sizes = np.array([r.system_size for r in successful_results])
        times = np.array([r.computation_time for r in successful_results])
        
        # Fit power law: t = a * N^b
        log_sizes = np.log(sizes)
        log_times = np.log(times)
        
        # Linear regression in log space
        coeffs = np.polyfit(log_sizes, log_times, 1)
        scaling_exponent = coeffs[0]
        log_prefactor = coeffs[1]
        prefactor = np.exp(log_prefactor)
        
        # Calculate R-squared
        predicted_log_times = np.polyval(coeffs, log_sizes)
        ss_res = np.sum((log_times - predicted_log_times) ** 2)
        ss_tot = np.sum((log_times - np.mean(log_times)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'scaling_exponent': scaling_exponent,
            'prefactor': prefactor,
            'r_squared': r_squared,
            'theoretical_exponent': 3.0,  # Expected N³ scaling for dense matrices
            'efficiency_ratio': scaling_exponent / 3.0
        }
    
    def generate_scaling_plot(self, save_path: Optional[str] = None) -> str:
        """
        Generate scaling analysis plot.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Path where plot was saved
        """
        if not self.results:
            raise ValueError("No performance results available.")
        
        successful_results = [r for r in self.results if r.convergence_achieved]
        
        sizes = [r.system_size for r in successful_results]
        times = [r.computation_time for r in successful_results]
        
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        plt.loglog(sizes, times, 'bo-', label='Measured', markersize=8)
        
        # Plot theoretical scaling lines
        if len(successful_results) >= 2:
            size_range = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
            
            # Fit line
            analysis = self.analyze_scaling_behavior()
            fitted_times = analysis['prefactor'] * (size_range ** analysis['scaling_exponent'])
            plt.loglog(size_range, fitted_times, 'r--', 
                      label=f'Fitted: N^{analysis["scaling_exponent"]:.2f}')
            
            # Theoretical N³ scaling
            theoretical_prefactor = times[0] / (sizes[0] ** 3)
            theoretical_times = theoretical_prefactor * (size_range ** 3)
            plt.loglog(size_range, theoretical_times, 'g:', 
                      label='Theoretical: N³')
        
        plt.xlabel('System Size (Number of Levels)')
        plt.ylabel('Computation Time (seconds)')
        plt.title('QBES Computational Scaling Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add performance summary text
        if len(successful_results) >= 3:
            analysis = self.analyze_scaling_behavior()
            textstr = f'Scaling Exponent: {analysis["scaling_exponent"]:.2f}\n'
            textstr += f'R²: {analysis["r_squared"]:.3f}\n'
            textstr += f'Efficiency: {analysis["efficiency_ratio"]:.2f}'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=props)
        
        if save_path is None:
            save_path = 'qbes_scaling_analysis.png'
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.results:
            return "No performance results available."
        
        successful_results = [r for r in self.results if r.convergence_achieved]
        failed_results = [r for r in self.results if not r.convergence_achieved]
        
        report = []
        report.append("=" * 60)
        report.append("QBES Performance Benchmark Report")
        report.append("=" * 60)
        report.append(f"Total Tests: {len(self.results)}")
        report.append(f"Successful: {len(successful_results)}")
        report.append(f"Failed: {len(failed_results)}")
        report.append("")
        
        if successful_results:
            # Performance summary
            sizes = [r.system_size for r in successful_results]
            times = [r.computation_time for r in successful_results]
            
            report.append("Performance Summary:")
            report.append("-" * 20)
            report.append(f"System Size Range: {min(sizes)} - {max(sizes)}")
            report.append(f"Time Range: {min(times):.3f}s - {max(times):.3f}s")
            report.append(f"Speedup Factor: {max(times)/min(times):.1f}x slower for largest system")
            report.append("")
            
            # Scaling analysis
            if len(successful_results) >= 3:
                try:
                    analysis = self.analyze_scaling_behavior()
                    report.append("Scaling Analysis:")
                    report.append("-" * 16)
                    report.append(f"Measured Scaling Exponent: {analysis['scaling_exponent']:.2f}")
                    report.append(f"Theoretical Exponent: {analysis['theoretical_exponent']:.1f}")
                    report.append(f"Efficiency Ratio: {analysis['efficiency_ratio']:.2f}")
                    report.append(f"Fit Quality (R²): {analysis['r_squared']:.3f}")
                    report.append("")
                    
                    # Performance interpretation
                    if analysis['scaling_exponent'] < 2.5:
                        performance_rating = "Excellent"
                    elif analysis['scaling_exponent'] < 3.5:
                        performance_rating = "Good"
                    elif analysis['scaling_exponent'] < 4.0:
                        performance_rating = "Acceptable"
                    else:
                        performance_rating = "Poor"
                    
                    report.append(f"Performance Rating: {performance_rating}")
                    report.append("")
                    
                except Exception as e:
                    report.append(f"Scaling analysis failed: {str(e)}")
                    report.append("")
        
        # Detailed results
        report.append("Detailed Results:")
        report.append("-" * 17)
        
        for result in self.results:
            status = "SUCCESS" if result.convergence_achieved else "FAILED"
            report.append(f"Size {result.system_size:2d}: {status} - {result.computation_time:.3f}s")
            
            if result.error_message:
                report.append(f"         Error: {result.error_message}")
        
        report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        report.append("-" * 15)
        
        if successful_results:
            max_successful_size = max(r.system_size for r in successful_results)
            report.append(f"Maximum recommended system size: {max_successful_size}")
            
            if len(successful_results) >= 3:
                analysis = self.analyze_scaling_behavior()
                if analysis['scaling_exponent'] > 3.5:
                    report.append("Consider optimizing matrix operations for better scaling.")
                if analysis['r_squared'] < 0.95:
                    report.append("Scaling behavior is irregular; investigate numerical stability.")
        
        if failed_results:
            min_failed_size = min(r.system_size for r in failed_results)
            report.append(f"System becomes unstable at size {min_failed_size}")
            report.append("Consider implementing adaptive algorithms for large systems.")
        
        return "\n".join(report)


def run_performance_benchmarks(max_size: int = 32) -> PerformanceBenchmarker:
    """
    Run comprehensive performance benchmarks.
    
    Args:
        max_size: Maximum system size to test
        
    Returns:
        PerformanceBenchmarker with results
    """
    print("Running QBES Performance Benchmarks")
    print("=" * 40)
    
    benchmarker = PerformanceBenchmarker()
    
    # Generate system sizes (powers of 2 up to max_size)
    system_sizes = [2**i for i in range(1, int(np.log2(max_size)) + 1)]
    
    # Run scaling tests
    benchmarker.test_system_size_scaling(system_sizes)
    
    # Generate report
    print("\nPerformance Analysis:")
    print(benchmarker.generate_performance_report())
    
    # Generate plot
    try:
        plot_path = benchmarker.generate_scaling_plot()
        print(f"\nScaling plot saved to: {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {str(e)}")
    
    return benchmarker