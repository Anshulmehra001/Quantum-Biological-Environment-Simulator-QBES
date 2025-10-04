"""
Automated benchmark execution and comparison system.

This module provides automated execution of benchmark suites
with comparison against reference results and automated reporting.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

from .benchmark_systems import BenchmarkRunner, BenchmarkResult
from .performance_benchmarks import PerformanceBenchmarker, run_performance_benchmarks


@dataclass
class BenchmarkSession:
    """Information about a benchmark session."""
    timestamp: str
    qbes_version: str
    system_info: Dict[str, Any]
    benchmark_results: List[Dict[str, Any]]
    performance_results: Optional[List[Dict[str, Any]]] = None
    session_id: str = ""


class AutomatedBenchmarkRunner:
    """
    Automated benchmark execution with result tracking and comparison.
    
    Provides functionality to run benchmark suites automatically,
    store results, and compare against historical data.
    """
    
    def __init__(self, results_dir: str = "benchmark_results"):
        """
        Initialize automated benchmark runner.
        
        Args:
            results_dir: Directory to store benchmark results
        """
        self.results_dir = results_dir
        self.ensure_results_directory()
    
    def ensure_results_directory(self):
        """Ensure results directory exists."""
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        import platform
        import sys
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_full_benchmark_suite(self, 
                                include_performance: bool = True,
                                max_performance_size: int = 16) -> BenchmarkSession:
        """
        Run complete benchmark suite with all tests.
        
        Args:
            include_performance: Whether to include performance benchmarks
            max_performance_size: Maximum system size for performance tests
            
        Returns:
            BenchmarkSession with all results
        """
        print("Starting Automated QBES Benchmark Suite")
        print("=" * 50)
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Run standard benchmarks
        print("1. Running Standard Benchmark Tests...")
        runner = BenchmarkRunner()
        runner.add_standard_benchmarks()
        benchmark_results = runner.run_all_benchmarks()
        
        # Convert results to serializable format
        benchmark_dicts = [asdict(result) for result in benchmark_results]
        
        performance_dicts = None
        if include_performance:
            print("\n2. Running Performance Benchmarks...")
            try:
                perf_benchmarker = run_performance_benchmarks(max_performance_size)
                performance_dicts = [asdict(result) for result in perf_benchmarker.results]
            except Exception as e:
                print(f"Performance benchmarks failed: {str(e)}")
                performance_dicts = []
        
        # Create session
        session = BenchmarkSession(
            timestamp=datetime.now().isoformat(),
            qbes_version="1.0.0",  # TODO: Get from package metadata
            system_info=self.get_system_info(),
            benchmark_results=benchmark_dicts,
            performance_results=performance_dicts,
            session_id=session_id
        )
        
        # Save results
        self.save_session(session)
        
        print(f"\n3. Benchmark session completed: {session_id}")
        print(f"Results saved to: {self.get_session_path(session_id)}")
        
        return session
    
    def save_session(self, session: BenchmarkSession):
        """Save benchmark session to file."""
        session_path = self.get_session_path(session.session_id)
        
        with open(session_path, 'w') as f:
            json.dump(asdict(session), f, indent=2)
    
    def load_session(self, session_id: str) -> BenchmarkSession:
        """Load benchmark session from file."""
        session_path = self.get_session_path(session_id)
        
        if not os.path.exists(session_path):
            raise FileNotFoundError(f"Session {session_id} not found")
        
        with open(session_path, 'r') as f:
            session_dict = json.load(f)
        
        return BenchmarkSession(**session_dict)
    
    def get_session_path(self, session_id: str) -> str:
        """Get file path for session."""
        return os.path.join(self.results_dir, f"benchmark_session_{session_id}.json")
    
    def list_sessions(self) -> List[str]:
        """List all available benchmark sessions."""
        sessions = []
        
        for filename in os.listdir(self.results_dir):
            if filename.startswith("benchmark_session_") and filename.endswith(".json"):
                session_id = filename.replace("benchmark_session_", "").replace(".json", "")
                sessions.append(session_id)
        
        return sorted(sessions)
    
    def compare_sessions(self, session_id_1: str, session_id_2: str) -> str:
        """
        Compare two benchmark sessions.
        
        Args:
            session_id_1: First session ID
            session_id_2: Second session ID
            
        Returns:
            Comparison report as string
        """
        session1 = self.load_session(session_id_1)
        session2 = self.load_session(session_id_2)
        
        report = []
        report.append("Benchmark Session Comparison")
        report.append("=" * 40)
        report.append(f"Session 1: {session_id_1} ({session1.timestamp})")
        report.append(f"Session 2: {session_id_2} ({session2.timestamp})")
        report.append("")
        
        # Compare benchmark results
        report.append("Standard Benchmark Comparison:")
        report.append("-" * 30)
        
        # Create lookup dictionaries
        results1 = {r['system_name']: r for r in session1.benchmark_results}
        results2 = {r['system_name']: r for r in session2.benchmark_results}
        
        common_tests = set(results1.keys()) & set(results2.keys())
        
        for test_name in sorted(common_tests):
            r1 = results1[test_name]
            r2 = results2[test_name]
            
            report.append(f"{test_name}:")
            
            # Compare pass/fail status
            status1 = "PASS" if r1['test_passed'] else "FAIL"
            status2 = "PASS" if r2['test_passed'] else "FAIL"
            report.append(f"  Status: {status1} → {status2}")
            
            # Compare errors if both passed
            if r1['test_passed'] and r2['test_passed']:
                error_change = r2['relative_error'] - r1['relative_error']
                error_pct = (error_change / r1['relative_error']) * 100 if r1['relative_error'] > 0 else 0
                report.append(f"  Relative Error: {r1['relative_error']:.2e} → {r2['relative_error']:.2e} ({error_pct:+.1f}%)")
                
                time_change = r2['computation_time'] - r1['computation_time']
                time_pct = (time_change / r1['computation_time']) * 100 if r1['computation_time'] > 0 else 0
                report.append(f"  Computation Time: {r1['computation_time']:.3f}s → {r2['computation_time']:.3f}s ({time_pct:+.1f}%)")
            
            report.append("")
        
        # Compare performance results if available
        if session1.performance_results and session2.performance_results:
            report.append("Performance Comparison:")
            report.append("-" * 20)
            
            perf1 = {r['system_size']: r for r in session1.performance_results}
            perf2 = {r['system_size']: r for r in session2.performance_results}
            
            common_sizes = set(perf1.keys()) & set(perf2.keys())
            
            for size in sorted(common_sizes):
                p1 = perf1[size]
                p2 = perf2[size]
                
                if p1['convergence_achieved'] and p2['convergence_achieved']:
                    time_change = p2['computation_time'] - p1['computation_time']
                    time_pct = (time_change / p1['computation_time']) * 100 if p1['computation_time'] > 0 else 0
                    report.append(f"Size {size}: {p1['computation_time']:.3f}s → {p2['computation_time']:.3f}s ({time_pct:+.1f}%)")
        
        return "\n".join(report)
    
    def generate_trend_analysis(self, num_sessions: int = 5) -> str:
        """
        Generate trend analysis from recent sessions.
        
        Args:
            num_sessions: Number of recent sessions to analyze
            
        Returns:
            Trend analysis report
        """
        sessions = self.list_sessions()
        
        if len(sessions) < 2:
            return "Not enough sessions for trend analysis (need at least 2)."
        
        # Get most recent sessions
        recent_sessions = sessions[-num_sessions:]
        
        report = []
        report.append("Benchmark Trend Analysis")
        report.append("=" * 30)
        report.append(f"Analyzing {len(recent_sessions)} recent sessions")
        report.append("")
        
        # Load sessions
        session_data = []
        for session_id in recent_sessions:
            try:
                session = self.load_session(session_id)
                session_data.append(session)
            except Exception as e:
                report.append(f"Could not load session {session_id}: {str(e)}")
        
        if len(session_data) < 2:
            return "\n".join(report + ["Not enough valid sessions for analysis."])
        
        # Analyze trends for each benchmark
        all_test_names = set()
        for session in session_data:
            for result in session.benchmark_results:
                all_test_names.add(result['system_name'])
        
        report.append("Test Reliability Trends:")
        report.append("-" * 25)
        
        for test_name in sorted(all_test_names):
            # Collect pass rates over time
            pass_rates = []
            timestamps = []
            
            for session in session_data:
                test_results = [r for r in session.benchmark_results if r['system_name'] == test_name]
                if test_results:
                    pass_rates.append(1.0 if test_results[0]['test_passed'] else 0.0)
                    timestamps.append(session.timestamp)
            
            if len(pass_rates) >= 2:
                recent_pass_rate = np.mean(pass_rates[-3:]) if len(pass_rates) >= 3 else pass_rates[-1]
                overall_pass_rate = np.mean(pass_rates)
                
                report.append(f"{test_name}:")
                report.append(f"  Overall Pass Rate: {overall_pass_rate:.1%}")
                report.append(f"  Recent Pass Rate: {recent_pass_rate:.1%}")
                
                if recent_pass_rate < overall_pass_rate:
                    report.append("  ⚠️  Declining reliability")
                elif recent_pass_rate > overall_pass_rate:
                    report.append("  ✅ Improving reliability")
                
                report.append("")
        
        return "\n".join(report)
    
    def create_benchmark_dashboard(self) -> str:
        """Create a summary dashboard of benchmark status."""
        sessions = self.list_sessions()
        
        if not sessions:
            return "No benchmark sessions found."
        
        # Get latest session
        latest_session = self.load_session(sessions[-1])
        
        report = []
        report.append("QBES Benchmark Dashboard")
        report.append("=" * 30)
        report.append(f"Latest Session: {latest_session.session_id}")
        report.append(f"Timestamp: {latest_session.timestamp}")
        report.append("")
        
        # Overall status
        total_tests = len(latest_session.benchmark_results)
        passed_tests = sum(1 for r in latest_session.benchmark_results if r['test_passed'])
        
        report.append("Current Status:")
        report.append("-" * 15)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {total_tests - passed_tests}")
        report.append(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        report.append("")
        
        # Test details
        report.append("Test Details:")
        report.append("-" * 13)
        
        for result in latest_session.benchmark_results:
            status = "✅ PASS" if result['test_passed'] else "❌ FAIL"
            report.append(f"{result['system_name']}: {status}")
            
            if result['test_passed']:
                report.append(f"  Error: {result['relative_error']:.2e}")
                report.append(f"  Time: {result['computation_time']:.3f}s")
            else:
                if result.get('error_message'):
                    report.append(f"  Error: {result['error_message']}")
        
        report.append("")
        
        # Historical summary
        if len(sessions) > 1:
            report.append("Historical Summary:")
            report.append("-" * 18)
            report.append(f"Total Sessions: {len(sessions)}")
            report.append(f"Date Range: {sessions[0]} to {sessions[-1]}")
            
            # Calculate average success rate
            success_rates = []
            for session_id in sessions[-5:]:  # Last 5 sessions
                try:
                    session = self.load_session(session_id)
                    total = len(session.benchmark_results)
                    passed = sum(1 for r in session.benchmark_results if r['test_passed'])
                    success_rates.append(passed / total if total > 0 else 0)
                except:
                    continue
            
            if success_rates:
                avg_success_rate = np.mean(success_rates)
                report.append(f"Average Success Rate (last 5): {avg_success_rate*100:.1f}%")
        
        return "\n".join(report)


def run_automated_benchmarks(results_dir: str = "benchmark_results",
                           include_performance: bool = True) -> AutomatedBenchmarkRunner:
    """
    Run automated benchmark suite with full reporting.
    
    Args:
        results_dir: Directory to store results
        include_performance: Whether to include performance benchmarks
        
    Returns:
        AutomatedBenchmarkRunner with session results
    """
    runner = AutomatedBenchmarkRunner(results_dir)
    
    # Run full benchmark suite
    session = runner.run_full_benchmark_suite(include_performance=include_performance)
    
    # Generate reports
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(runner.create_benchmark_dashboard())
    
    # Show trend analysis if multiple sessions exist
    sessions = runner.list_sessions()
    if len(sessions) > 1:
        print("\n" + "="*60)
        print("TREND ANALYSIS")
        print("="*60)
        print(runner.generate_trend_analysis())
    
    return runner