#!/usr/bin/env python3
"""
Performance Analysis and Stress Testing for QBES

This script performs performance analysis and stress testing to evaluate
system scalability and resource usage.
"""

import sys
import os
import time
import json
import psutil
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

# Add the qbes package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class PerformanceAnalyzer:
    """Performance analysis and stress testing for QBES."""
    
    def __init__(self, results_dir="performance_results"):
        """Initialize the performance analyzer."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.performance_data = []
        self.start_time = datetime.now()
        
    def measure_system_resources(self):
        """Measure current system resource usage."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('.').percent
        }
    
    def test_quantum_system_scaling(self):
        """Test performance scaling with quantum system size."""
        print("\n1. Testing Quantum System Scaling...")
        print("-" * 40)
        
        system_sizes = [2, 4, 6, 8, 10, 12]  # Number of qubits
        scaling_results = []
        
        for n_qubits in system_sizes:
            print(f"Testing {n_qubits}-qubit system...")
            
            test_start = time.time()
            initial_resources = self.measure_system_resources()
            
            try:
                from qbes.quantum_engine import QuantumEngine
                from qbes.core.data_models import QuantumState
                
                # Create quantum engine
                engine = QuantumEngine()
                
                # Initialize quantum state for n_qubits
                state = engine.initialize_state(n_qubits=n_qubits)
                
                # Perform some quantum operations
                for _ in range(10):  # 10 time steps
                    state = engine.evolve_state(state, time_step=0.01)
                
                duration = time.time() - test_start
                final_resources = self.measure_system_resources()
                
                result = {
                    'n_qubits': n_qubits,
                    'duration': duration,
                    'success': True,
                    'initial_memory': initial_resources['memory_percent'],
                    'final_memory': final_resources['memory_percent'],
                    'memory_increase': final_resources['memory_percent'] - initial_resources['memory_percent']
                }
                
                print(f"  {n_qubits} qubits: {duration:.2f}s, Memory: +{result['memory_increase']:.1f}%")
                
            except Exception as e:
                duration = time.time() - test_start
                result = {
                    'n_qubits': n_qubits,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                }
                print(f"  {n_qubits} qubits: FAILED ({duration:.2f}s) - {str(e)}")
            
            scaling_results.append(result)
            
            # Brief pause between tests
            time.sleep(1)
        
        self.performance_data.append({
            'test_name': 'Quantum System Scaling',
            'results': scaling_results,
            'timestamp': datetime.now().isoformat()
        })
        
        return scaling_results
    
    def test_simulation_duration_scaling(self):
        """Test performance scaling with simulation duration."""
        print("\n2. Testing Simulation Duration Scaling...")
        print("-" * 40)
        
        durations = [0.1, 0.5, 1.0, 2.0, 5.0]  # Simulation times in seconds
        duration_results = []
        
        for sim_time in durations:
            print(f"Testing {sim_time}s simulation...")
            
            test_start = time.time()
            initial_resources = self.measure_system_resources()
            
            try:
                from qbes.simulation_engine import SimulationEngine
                from qbes.core.data_models import SimulationConfig
                
                # Create test configuration
                config = SimulationConfig(
                    system_pdb="test_two_level.pdb",
                    temperature=300.0,
                    simulation_time=sim_time,
                    time_step=0.01,
                    quantum_subsystem_selection="all",
                    noise_model_type="ohmic",
                    output_directory="perf_test_output"
                )
                
                # Run simulation
                engine = SimulationEngine()
                results = engine.run_simulation(config)
                
                duration = time.time() - test_start
                final_resources = self.measure_system_resources()
                
                result = {
                    'simulation_time': sim_time,
                    'wall_clock_time': duration,
                    'success': True,
                    'initial_memory': initial_resources['memory_percent'],
                    'final_memory': final_resources['memory_percent'],
                    'efficiency_ratio': sim_time / duration if duration > 0 else 0
                }
                
                print(f"  {sim_time}s sim: {duration:.2f}s wall time, Efficiency: {result['efficiency_ratio']:.3f}")
                
            except Exception as e:
                duration = time.time() - test_start
                result = {
                    'simulation_time': sim_time,
                    'wall_clock_time': duration,
                    'success': False,
                    'error': str(e)
                }
                print(f"  {sim_time}s sim: FAILED ({duration:.2f}s) - {str(e)}")
            
            duration_results.append(result)
            
            # Brief pause between tests
            time.sleep(1)
        
        self.performance_data.append({
            'test_name': 'Simulation Duration Scaling',
            'results': duration_results,
            'timestamp': datetime.now().isoformat()
        })
        
        return duration_results
    
    def test_memory_stress(self):
        """Test system behavior under memory stress."""
        print("\n3. Testing Memory Stress...")
        print("-" * 40)
        
        # Test with increasingly large data structures
        memory_results = []
        
        test_sizes = [100, 500, 1000, 2000, 5000]  # Array sizes
        
        for size in test_sizes:
            print(f"Testing with {size}x{size} matrices...")
            
            test_start = time.time()
            initial_resources = self.measure_system_resources()
            
            try:
                # Create large matrices to stress memory
                matrices = []
                for i in range(10):  # Create 10 large matrices
                    matrix = np.random.complex128((size, size))
                    matrices.append(matrix)
                
                # Perform some operations
                result_matrix = matrices[0]
                for matrix in matrices[1:]:
                    result_matrix = result_matrix @ matrix[:size, :size]
                
                duration = time.time() - test_start
                final_resources = self.measure_system_resources()
                
                result = {
                    'matrix_size': size,
                    'duration': duration,
                    'success': True,
                    'initial_memory': initial_resources['memory_percent'],
                    'final_memory': final_resources['memory_percent'],
                    'peak_memory_usage': final_resources['memory_percent']
                }
                
                print(f"  {size}x{size}: {duration:.2f}s, Peak Memory: {result['peak_memory_usage']:.1f}%")
                
                # Clean up
                del matrices, result_matrix
                
            except Exception as e:
                duration = time.time() - test_start
                result = {
                    'matrix_size': size,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                }
                print(f"  {size}x{size}: FAILED ({duration:.2f}s) - {str(e)}")
            
            memory_results.append(result)
            
            # Force garbage collection and pause
            import gc
            gc.collect()
            time.sleep(2)
        
        self.performance_data.append({
            'test_name': 'Memory Stress Test',
            'results': memory_results,
            'timestamp': datetime.now().isoformat()
        })
        
        return memory_results
    
    def test_concurrent_operations(self):
        """Test performance with concurrent operations."""
        print("\n4. Testing Concurrent Operations...")
        print("-" * 40)
        
        import threading
        import queue
        
        concurrent_results = []
        thread_counts = [1, 2, 4, 8]
        
        for n_threads in thread_counts:
            print(f"Testing with {n_threads} concurrent threads...")
            
            test_start = time.time()
            initial_resources = self.measure_system_resources()
            
            try:
                # Create a queue for results
                result_queue = queue.Queue()
                
                def worker_function(worker_id):
                    """Worker function for concurrent testing."""
                    try:
                        from qbes.quantum_engine import QuantumEngine
                        
                        engine = QuantumEngine()
                        state = engine.initialize_state(n_qubits=4)
                        
                        # Perform some operations
                        for _ in range(5):
                            state = engine.evolve_state(state, time_step=0.01)
                        
                        result_queue.put({'worker_id': worker_id, 'success': True})
                        
                    except Exception as e:
                        result_queue.put({'worker_id': worker_id, 'success': False, 'error': str(e)})
                
                # Create and start threads
                threads = []
                for i in range(n_threads):
                    thread = threading.Thread(target=worker_function, args=(i,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                
                # Collect results
                worker_results = []
                while not result_queue.empty():
                    worker_results.append(result_queue.get())
                
                duration = time.time() - test_start
                final_resources = self.measure_system_resources()
                
                successful_workers = sum(1 for r in worker_results if r['success'])
                
                result = {
                    'n_threads': n_threads,
                    'duration': duration,
                    'successful_workers': successful_workers,
                    'total_workers': n_threads,
                    'success_rate': successful_workers / n_threads,
                    'initial_memory': initial_resources['memory_percent'],
                    'final_memory': final_resources['memory_percent']
                }
                
                print(f"  {n_threads} threads: {duration:.2f}s, Success: {successful_workers}/{n_threads}")
                
            except Exception as e:
                duration = time.time() - test_start
                result = {
                    'n_threads': n_threads,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                }
                print(f"  {n_threads} threads: FAILED ({duration:.2f}s) - {str(e)}")
            
            concurrent_results.append(result)
            
            # Brief pause between tests
            time.sleep(2)
        
        self.performance_data.append({
            'test_name': 'Concurrent Operations',
            'results': concurrent_results,
            'timestamp': datetime.now().isoformat()
        })
        
        return concurrent_results
    
    def analyze_scaling_behavior(self, scaling_results):
        """Analyze scaling behavior from test results."""
        print("\n5. Analyzing Scaling Behavior...")
        print("-" * 40)
        
        successful_results = [r for r in scaling_results if r['success']]
        
        if len(successful_results) < 2:
            print("Insufficient data for scaling analysis.")
            return None
        
        # Extract data for analysis
        sizes = np.array([r['n_qubits'] for r in successful_results])
        times = np.array([r['duration'] for r in successful_results])
        
        # Fit scaling law: time = a * size^b
        log_sizes = np.log(sizes)
        log_times = np.log(times)
        
        # Linear fit in log space
        coeffs = np.polyfit(log_sizes, log_times, 1)
        scaling_exponent = coeffs[0]
        scaling_constant = np.exp(coeffs[1])
        
        # Calculate R-squared
        predicted_log_times = np.polyval(coeffs, log_sizes)
        ss_res = np.sum((log_times - predicted_log_times) ** 2)
        ss_tot = np.sum((log_times - np.mean(log_times)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        analysis = {
            'scaling_exponent': scaling_exponent,
            'scaling_constant': scaling_constant,
            'r_squared': r_squared,
            'complexity_class': self.classify_complexity(scaling_exponent)
        }
        
        print(f"Scaling Analysis:")
        print(f"  Scaling exponent: {scaling_exponent:.2f}")
        print(f"  R-squared: {r_squared:.3f}")
        print(f"  Complexity class: {analysis['complexity_class']}")
        
        return analysis
    
    def classify_complexity(self, exponent):
        """Classify computational complexity based on scaling exponent."""
        if exponent < 1.5:
            return "Linear/Sub-quadratic"
        elif exponent < 2.5:
            return "Quadratic"
        elif exponent < 3.5:
            return "Cubic"
        elif exponent < 4.5:
            return "Quartic"
        else:
            return "Higher-order polynomial/Exponential"
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "=" * 60)
        print("GENERATING PERFORMANCE REPORT")
        print("=" * 60)
        
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        # Generate detailed report
        report = {
            'performance_summary': {
                'total_test_duration': total_duration,
                'test_start_time': self.start_time.isoformat(),
                'test_end_time': datetime.now().isoformat(),
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            },
            'performance_data': self.performance_data
        }
        
        # Save JSON report
        json_report_path = self.results_dir / "performance_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable report
        text_report_path = self.results_dir / "performance_report.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("QBES Performance Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test Execution Summary:\n")
            f.write(f"  Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Total Duration: {total_duration:.1f} seconds\n")
            f.write(f"  CPU Cores: {psutil.cpu_count()}\n")
            f.write(f"  Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB\n\n")
            
            f.write("Performance Test Results:\n")
            f.write("-" * 30 + "\n")
            
            for test_data in self.performance_data:
                f.write(f"\n{test_data['test_name']}:\n")
                
                successful_tests = [r for r in test_data['results'] if r.get('success', False)]
                total_tests = len(test_data['results'])
                
                f.write(f"  Total Tests: {total_tests}\n")
                f.write(f"  Successful: {len(successful_tests)}\n")
                f.write(f"  Success Rate: {len(successful_tests)/total_tests:.1%}\n")
                
                if successful_tests:
                    durations = [r['duration'] for r in successful_tests]
                    f.write(f"  Average Duration: {np.mean(durations):.2f}s\n")
                    f.write(f"  Min Duration: {min(durations):.2f}s\n")
                    f.write(f"  Max Duration: {max(durations):.2f}s\n")
        
        print(f"\nPerformance reports generated:")
        print(f"  JSON Report: {json_report_path}")
        print(f"  Text Report: {text_report_path}")
        
        return report
    
    def run_all_performance_tests(self):
        """Run all performance tests."""
        print("QBES Performance Analysis and Stress Testing")
        print("=" * 60)
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results Directory: {self.results_dir}")
        
        # Run all performance tests
        scaling_results = self.test_quantum_system_scaling()
        duration_results = self.test_simulation_duration_scaling()
        memory_results = self.test_memory_stress()
        concurrent_results = self.test_concurrent_operations()
        
        # Analyze scaling behavior
        scaling_analysis = self.analyze_scaling_behavior(scaling_results)
        
        # Generate comprehensive report
        report = self.generate_performance_report()
        
        return report, scaling_analysis


def main():
    """Main function to run performance analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run QBES performance analysis and stress testing"
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='performance_results',
        help='Directory to store performance results'
    )
    
    args = parser.parse_args()
    
    try:
        # Run performance analysis
        analyzer = PerformanceAnalyzer(results_dir=args.results_dir)
        report, scaling_analysis = analyzer.run_all_performance_tests()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        summary = report['performance_summary']
        print(f"Total Test Duration: {summary['total_test_duration']:.1f} seconds")
        print(f"System: {summary['system_info']['cpu_count']} cores, "
              f"{summary['system_info']['memory_total_gb']:.1f} GB RAM")
        
        if scaling_analysis:
            print(f"Scaling Behavior: {scaling_analysis['complexity_class']}")
            print(f"Scaling Exponent: {scaling_analysis['scaling_exponent']:.2f}")
        
        print("\nPerformance analysis completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nPerformance analysis interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nPerformance analysis failed: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())