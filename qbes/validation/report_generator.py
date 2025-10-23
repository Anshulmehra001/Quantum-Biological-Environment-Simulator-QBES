"""
Validation Report Generator for QBES

This module provides comprehensive report generation capabilities for validation
results, including detailed analysis, performance benchmarking, and scientific
interpretation of results.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from .accuracy_calculator import AccuracyResult, AccuracyCalculator

logger = logging.getLogger(__name__)


class ValidationReportGenerator:
    """
    Generates comprehensive validation reports with detailed analysis.
    
    This class creates markdown reports containing test results, accuracy metrics,
    performance benchmarking, computational efficiency analysis, and scientific
    interpretation with recommendations.
    """
    
    def __init__(self, output_dir: str = "validation_reports"):
        """
        Initialize the ValidationReportGenerator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def generate_report(self, 
                       validation_results: Dict[str, Any],
                       performance_data: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            validation_results: Results from accuracy calculator
            performance_data: Performance benchmarking data
            metadata: Additional metadata about the validation run
            
        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"validation_report_{timestamp}.md"
        report_path = self.output_dir / report_filename
        
        # Generate report content
        report_content = self._generate_report_content(
            validation_results, performance_data, metadata
        )
        
        # Write report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Validation report generated: {report_path}")
        return str(report_path)
    
    def _generate_report_content(self, 
                                validation_results: Dict[str, Any],
                                pe