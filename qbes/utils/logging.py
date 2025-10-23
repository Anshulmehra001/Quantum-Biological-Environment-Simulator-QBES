"""
Enhanced logging utilities for QBES with progress indicators and verbose output.
"""

import time
import sys
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class LogLevel(Enum):
    """Logging levels for QBES."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


@dataclass
class ProgressInfo:
    """Information about simulation progress."""
    current_step: int
    total_steps: int
    current_time: float
    target_time: float
    start_time: float
    phase: str = "simulation"
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100.0)
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def estimated_total_time(self) -> float:
        """Estimate total simulation time."""
        if self.progress_percent == 0:
            return 0.0
        return self.elapsed_time * 100.0 / self.progress_percent
    
    @property
    def estimated_remaining_time(self) -> float:
        """Estimate remaining simulation time."""
        return max(0.0, self.estimated_total_time - self.elapsed_time)


class EnhancedLogger:
    """Enhanced logger with progress indicators and verbose output."""
    
    def __init__(self, verbose: bool = False, log_level: LogLevel = LogLevel.INFO):
        self.verbose = verbose
        self.log_level = log_level
        self.current_progress: Optional[ProgressInfo] = None
        self.last_progress_update = 0.0
        self.progress_update_interval = 1.0  # seconds
        
    def set_verbose(self, verbose: bool):
        """Set verbose logging mode."""
        self.verbose = verbose
    
    def set_log_level(self, level: LogLevel):
        """Set minimum logging level."""
        self.log_level = level
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO, end: str = '\n'):
        """Log a message with specified level."""
        if level.value >= self.log_level.value:
            timestamp = time.strftime("%H:%M:%S")
            level_str = level.name.ljust(7)
            
            if self.verbose or level.value >= LogLevel.WARNING.value:
                print(f"[{timestamp}] {level_str} {message}", end=end, flush=True)
            elif level == LogLevel.INFO:
                print(message, end=end, flush=True)
    
    def debug(self, message: str):
        """Log debug message."""
        self.log(message, LogLevel.DEBUG)
    
    def info(self, message: str):
        """Log info message."""
        self.log(message, LogLevel.INFO)
    
    def warning(self, message: str):
        """Log warning message."""
        self.log(message, LogLevel.WARNING)
    
    def error(self, message: str):
        """Log error message."""
        self.log(message, LogLevel.ERROR)
    
    def log_simulation_step(self, step: str, progress: Optional[ProgressInfo] = None):
        """Log a major simulation step with clear status message."""
        step_messages = {
            'md_initialization': '🔧 Initializing molecular dynamics system...',
            'quantum_setup': '⚛️  Setting up quantum subsystem...',
            'hamiltonian_construction': '🧮 Constructing system Hamiltonian...',
            'noise_model_setup': '🌊 Configuring environmental noise model...',
            'quantum_evolution': '🚀 Starting quantum evolution...',
            'md_evolution': '🔄 Running molecular dynamics...',
            'coupling_calculation': '🔗 Calculating quantum-classical coupling...',
            'analysis': '📊 Analyzing simulation results...',
            'output_generation': '💾 Generating output files...',
            'completion': '✅ Simulation completed successfully!'
        }
        
        message = step_messages.get(step, f"Processing: {step}")
        self.info(message)
        
        if progress:
            self.current_progress = progress
            self.update_progress_display()
    
    def update_progress_display(self, force_update: bool = False):
        """Update progress display if enough time has passed."""
        if not self.current_progress:
            return
        
        current_time = time.time()
        if not force_update and (current_time - self.last_progress_update) < self.progress_update_interval:
            return
        
        progress = self.current_progress
        
        # Create progress bar
        bar_width = 30
        filled_width = int(bar_width * progress.progress_percent / 100.0)
        bar = '█' * filled_width + '░' * (bar_width - filled_width)
        
        # Format time displays
        elapsed_str = self._format_time(progress.elapsed_time)
        remaining_str = self._format_time(progress.estimated_remaining_time)
        
        # Create progress line
        progress_line = (
            f"\r  Progress: [{bar}] {progress.progress_percent:5.1f}% | "
            f"Step: {progress.current_step:,}/{progress.total_steps:,} | "
            f"Elapsed: {elapsed_str} | "
            f"ETA: {remaining_str}"
        )
        
        print(progress_line, end='', flush=True)
        self.last_progress_update = current_time
    
    def finish_progress(self):
        """Finish progress display and move to new line."""
        if self.current_progress:
            self.update_progress_display(force_update=True)
            print()  # New line
            self.current_progress = None
    
    def print_results_summary(self, results):
        """Print formatted results table to terminal."""
        from ..core.data_models import SimulationResults
        
        if not isinstance(results, SimulationResults):
            self.error("Invalid results object for summary display")
            return
        
        self.info("\n" + "=" * 60)
        self.info("SIMULATION RESULTS SUMMARY")
        self.info("=" * 60)
        
        # Extract key metrics
        try:
            # Get final state metrics
            final_state = results.state_trajectory[-1] if results.state_trajectory else None
            final_energy = results.energy_trajectory[-1] if results.energy_trajectory else None
            
            # Calculate coherence lifetime if available
            coherence_lifetime = None
            if 'coherence' in results.coherence_measures:
                coherence_data = results.coherence_measures['coherence']
                if coherence_data:
                    # Simple exponential decay fit approximation
                    coherence_lifetime = self._estimate_coherence_lifetime(coherence_data)
            
            # Calculate purity
            purity = None
            if final_state:
                eigenvals = abs(final_state.matrix.diagonal())
                purity = sum(p**2 for p in eigenvals)
            
            # Decoherence rates
            decoherence_rate = None
            if results.decoherence_rates:
                decoherence_rate = results.decoherence_rates.get('total', 
                                 list(results.decoherence_rates.values())[0])
            
            # Create summary table
            table_data = [
                ("Simulation Time", f"{results.simulation_config.simulation_time:.2e} s"),
                ("Final Energy", f"{final_energy:.6f} a.u." if final_energy else "N/A"),
                ("Final Purity", f"{purity:.4f}" if purity else "N/A"),
                ("Coherence Lifetime", f"{coherence_lifetime:.2e} s" if coherence_lifetime else "N/A"),
                ("Decoherence Rate", f"{decoherence_rate:.2e} s⁻¹" if decoherence_rate else "N/A"),
                ("Computation Time", f"{results.computation_time:.2f} s"),
                ("Output Directory", results.simulation_config.output_directory)
            ]
            
            # Print table
            max_label_width = max(len(label) for label, _ in table_data)
            for label, value in table_data:
                self.info(f"  {label:<{max_label_width}} : {value}")
            
            self.info("=" * 60)
            
            # Additional statistics if available
            if results.statistical_summary:
                self.info("\nSTATISTICAL SUMMARY")
                self.info("-" * 20)
                
                stats = results.statistical_summary
                for metric, mean_val in stats.mean_values.items():
                    std_val = stats.std_deviations.get(metric, 0.0)
                    self.info(f"  {metric}: {mean_val:.4e} ± {std_val:.4e}")
                
                self.info(f"  Sample Size: {stats.sample_size}")
            
        except Exception as e:
            self.error(f"Error generating results summary: {e}")
            # Fallback to basic info
            self.info(f"  Simulation completed in {results.computation_time:.2f} seconds")
            self.info(f"  Results saved to: {results.simulation_config.output_directory}")
    
    def format_error_message(self, error: Exception, context: str = "") -> str:
        """Format helpful error messages with suggestions."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Common error patterns and suggestions
        suggestions = []
        
        if "FileNotFoundError" in error_type or "No such file" in error_msg:
            suggestions.extend([
                "• Check that the file path is correct",
                "• Ensure the file exists in the specified location",
                "• Use absolute paths if relative paths aren't working"
            ])
        
        elif "PermissionError" in error_type or "Permission denied" in error_msg:
            suggestions.extend([
                "• Check file/directory permissions",
                "• Ensure you have write access to the output directory",
                "• Try running with appropriate permissions"
            ])
        
        elif "ValueError" in error_type:
            if "positive" in error_msg.lower():
                suggestions.append("• Ensure all numerical parameters are positive values")
            elif "normalized" in error_msg.lower():
                suggestions.append("• Check quantum state normalization")
            else:
                suggestions.append("• Verify input parameter values and formats")
        
        elif "MemoryError" in error_type:
            suggestions.extend([
                "• Reduce system size or simulation time",
                "• Increase available system memory",
                "• Use checkpointing to reduce memory usage"
            ])
        
        elif "ImportError" in error_type or "ModuleNotFoundError" in error_type:
            suggestions.extend([
                "• Ensure all required dependencies are installed",
                "• Check your Python environment setup",
                "• Try reinstalling QBES and its dependencies"
            ])
        
        # Format the complete error message
        formatted_msg = f"\n{'='*60}\n"
        formatted_msg += f"ERROR: {error_type}\n"
        formatted_msg += f"{'='*60}\n"
        formatted_msg += f"{error_msg}\n"
        
        if context:
            formatted_msg += f"\nContext: {context}\n"
        
        if suggestions:
            formatted_msg += f"\nSuggestions:\n"
            for suggestion in suggestions:
                formatted_msg += f"{suggestion}\n"
        
        formatted_msg += f"{'='*60}\n"
        
        return formatted_msg
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def _estimate_coherence_lifetime(self, coherence_data: list) -> Optional[float]:
        """Estimate coherence lifetime from coherence data."""
        try:
            import numpy as np
            
            if len(coherence_data) < 2:
                return None
            
            # Simple exponential decay estimation
            # Assume coherence_data contains coherence values over time
            coherence_array = np.array(coherence_data)
            
            # Find 1/e point
            initial_coherence = coherence_array[0]
            target_coherence = initial_coherence / np.e
            
            # Find first point below target
            below_target = np.where(coherence_array <= target_coherence)[0]
            
            if len(below_target) > 0:
                # Estimate based on time step and position
                time_step = 1e-15  # Default femtosecond time step
                lifetime_steps = below_target[0]
                return lifetime_steps * time_step
            
            return None
            
        except Exception:
            return None


# Global logger instance
_global_logger: Optional[EnhancedLogger] = None


def get_logger(verbose: bool = False) -> EnhancedLogger:
    """Get or create global logger instance."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = EnhancedLogger(verbose=verbose)
    else:
        _global_logger.set_verbose(verbose)
    
    return _global_logger


def log_simulation_phase(phase: str, progress_info: Optional[ProgressInfo] = None):
    """Convenience function to log simulation phases."""
    logger = get_logger()
    logger.log_simulation_step(phase, progress_info)


def update_progress(current_step: int, total_steps: int, current_time: float = 0.0, 
                   target_time: float = 1.0, phase: str = "simulation"):
    """Convenience function to update progress."""
    logger = get_logger()
    
    if logger.current_progress:
        logger.current_progress.current_step = current_step
        logger.current_progress.total_steps = total_steps
        logger.current_progress.current_time = current_time
        logger.current_progress.target_time = target_time
        logger.current_progress.phase = phase
        logger.update_progress_display()
    else:
        # Create new progress info
        progress_info = ProgressInfo(
            current_step=current_step,
            total_steps=total_steps,
            current_time=current_time,
            target_time=target_time,
            start_time=time.time(),
            phase=phase
        )
        logger.current_progress = progress_info
        logger.update_progress_display()


def finish_progress():
    """Convenience function to finish progress display."""
    logger = get_logger()
    logger.finish_progress()