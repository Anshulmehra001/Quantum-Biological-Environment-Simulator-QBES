"""
Tests for QBES Debugging Loop with Automatic Error Correction

This module tests the debugging loop functionality including error detection,
automatic fix application, changelog updates, and the complete debugging workflow.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from qbes.validation.debugging_loop import (
    ErrorDetectionSystem, AutomaticFixSystem, ChangelogUpdater, DebuggingLoop,
    ErrorType, ErrorDiagnosis, FixAttempt, DebuggingSession
)
from qbes.validation.validator import ValidationSummary, QBESValidator


class TestErrorDetectionSystem:
    """Test error detection and diagnosis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_detector = ErrorDetectionSystem()
    
    def test_initialization(self):
        """Test ErrorDetectionSystem initialization."""
        assert self.error_detector is not None
        assert hasattr(self.error_detector, 'error_patterns')
        assert hasattr(self.error_detector, 'fix_templates')
        assert len(self.error_detector.error_patterns) > 0
        assert len(self.error_detector.fix_templates) > 0
    
    def test_error_pattern_definitions(self):
        """Test that error patterns are properly defined."""
        patterns = self.error_detector.error_patterns
        
        # Check that all error types have patterns
        expected_types = [
            ErrorType.NUMERICAL_PRECISION,
            ErrorType.ALGORITHM_IMPLEMENTATION,
            ErrorType.PARAMETER_CONFIGURATION,
            ErrorType.TOLERANCE_THRESHOLD,
            ErrorType.SYSTEM_INITIALIZATION,
            ErrorType.CONVERGENCE_ISSUE
        ]
        
        for error_type in expected_types:
            assert error_type in patterns
            assert len(patterns[error_type]) > 0
    
    def test_fix_template_definitions(self):
        """Test that fix templates are properly defined."""
        templates = self.error_detector.fix_templates
        
        # Check that all error types have fix templates
        for error_type in ErrorType:
            if error_type != ErrorType.UNKNOWN:
                assert error_type in templates
                assert len(templates[error_type]) > 0
    
    def test_classify_error_type(self):
        """Test error type classification."""
        # Test numerical precision classification
        assert self.error_detector._classify_error_type("numerical_precision_test") == ErrorType.NUMERICAL_PRECISION
        assert self.error_detector._classify_error_type("precision_loss_test") == ErrorType.NUMERICAL_PRECISION
        
        # Test algorithm implementation classification
        assert self.error_detector._classify_error_type("hamiltonian_construction") == ErrorType.ALGORITHM_IMPLEMENTATION
        assert self.error_detector._classify_error_type("evolution_step_test") == ErrorType.ALGORITHM_IMPLEMENTATION
        
        # Test tolerance threshold classification
        assert self.error_detector._classify_error_type("tolerance_check") == ErrorType.TOLERANCE_THRESHOLD
        assert self.error_detector._classify_error_type("threshold_validation") == ErrorType.TOLERANCE_THRESHOLD
        
        # Test unknown classification
        assert self.error_detector._classify_error_type("unknown_test") == ErrorType.UNKNOWN
    
    def test_detect_errors_with_failed_tests(self):
        """Test error detection with failed tests."""
        # Create mock validation summary with failed tests
        validation_summary = Mock(spec=ValidationSummary)
        validation_summary.failed_tests = ["numerical_precision_test", "tolerance_check"]
        validation_summary.final_accuracy = 95.0
        validation_summary.retry_history = [{"attempt": 1}]
        
        diagnoses = self.error_detector.detect_errors(validation_summary)
        
        assert len(diagnoses) >= 2  # At least one for each failed test
        
        # Check that diagnoses are properly formed
        for diagnosis in diagnoses:
            assert isinstance(diagnosis, ErrorDiagnosis)
            assert diagnosis.test_name in validation_summary.failed_tests or diagnosis.test_name == "overall_accuracy"
            assert diagnosis.confidence > 0.0
            assert diagnosis.fix_priority > 0
    
    def test_detect_errors_with_low_accuracy(self):
        """Test error detection with low overall accuracy."""
        validation_summary = Mock(spec=ValidationSummary)
        validation_summary.failed_tests = []
        validation_summary.final_accuracy = 85.0  # Below 98% threshold
        validation_summary.retry_history = [{"attempt": 1}]
        
        diagnoses = self.error_detector.detect_errors(validation_summary)
        
        # Should detect accuracy issue
        accuracy_diagnoses = [d for d in diagnoses if d.test_name == "overall_accuracy"]
        assert len(accuracy_diagnoses) > 0
        
        accuracy_diagnosis = accuracy_diagnoses[0]
        assert accuracy_diagnosis.error_type == ErrorType.ALGORITHM_IMPLEMENTATION
        assert accuracy_diagnosis.confidence > 0.0
    
    def test_detect_errors_with_retry_patterns(self):
        """Test error detection with multiple retry attempts."""
        validation_summary = Mock(spec=ValidationSummary)
        validation_summary.failed_tests = []
        validation_summary.final_accuracy = 99.0
        validation_summary.retry_history = [
            {"attempt": 1}, {"attempt": 2}, {"attempt": 3}
        ]
        
        diagnoses = self.error_detector.detect_errors(validation_summary)
        
        # Should detect retry pattern issue
        retry_diagnoses = [d for d in diagnoses if d.test_name == "retry_pattern"]
        assert len(retry_diagnoses) > 0
        
        retry_diagnosis = retry_diagnoses[0]
        assert retry_diagnosis.error_type == ErrorType.CONVERGENCE_ISSUE
    
    def test_diagnose_test_failure_numerical_precision(self):
        """Test diagnosis of numerical precision failures."""
        validation_summary = Mock(spec=ValidationSummary)
        
        diagnosis = self.error_detector._diagnose_test_failure("precision_test", validation_summary)
        
        assert diagnosis is not None
        assert diagnosis.error_type == ErrorType.NUMERICAL_PRECISION
        assert diagnosis.test_name == "precision_test"
        assert "precision" in diagnosis.description.lower()
        assert diagnosis.confidence > 0.0
        assert len(diagnosis.affected_files) > 0
    
    def test_diagnose_test_failure_algorithm_implementation(self):
        """Test diagnosis of algorithm implementation failures."""
        validation_summary = Mock(spec=ValidationSummary)
        
        diagnosis = self.error_detector._diagnose_test_failure("hamiltonian_test", validation_summary)
        
        assert diagnosis is not None
        assert diagnosis.error_type == ErrorType.ALGORITHM_IMPLEMENTATION
        assert diagnosis.test_name == "hamiltonian_test"
        assert "algorithm" in diagnosis.description.lower()
    
    def test_diagnose_test_failure_tolerance_threshold(self):
        """Test diagnosis of tolerance threshold failures."""
        validation_summary = Mock(spec=ValidationSummary)
        
        diagnosis = self.error_detector._diagnose_test_failure("tolerance_test", validation_summary)
        
        assert diagnosis is not None
        assert diagnosis.error_type == ErrorType.TOLERANCE_THRESHOLD
        assert diagnosis.test_name == "tolerance_test"
        assert "tolerance" in diagnosis.description.lower()


class TestAutomaticFixSystem:
    """Test automatic fix application functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fix_system = AutomaticFixSystem()
    
    def test_initialization(self):
        """Test AutomaticFixSystem initialization."""
        assert self.fix_system is not None
        assert hasattr(self.fix_system, 'applied_fixes')
        assert len(self.fix_system.applied_fixes) == 0
    
    def test_apply_fix_tolerance_threshold(self):
        """Test applying tolerance threshold fixes."""
        diagnosis = ErrorDiagnosis(
            error_type=ErrorType.TOLERANCE_THRESHOLD,
            test_name="tolerance_test",
            description="Tolerance too strict",
            root_cause="Overly strict thresholds",
            suggested_fix="Relax tolerances",
            confidence=0.9,
            affected_files=["reference_data.json"],
            fix_priority=1
        )
        
        fix_attempt = self.fix_system.apply_fix(diagnosis)
        
        assert isinstance(fix_attempt, FixAttempt)
        assert fix_attempt.diagnosis == diagnosis
        assert fix_attempt.success is True
        assert len(fix_attempt.fix_applied) > 0
        assert "tolerance" in fix_attempt.fix_applied.lower()
        assert len(self.fix_system.applied_fixes) == 1
    
    def test_apply_fix_numerical_precision(self):
        """Test applying numerical precision fixes."""
        diagnosis = ErrorDiagnosis(
            error_type=ErrorType.NUMERICAL_PRECISION,
            test_name="precision_test",
            description="Precision issues",
            root_cause="Insufficient precision",
            suggested_fix="Increase precision",
            confidence=0.8,
            affected_files=["quantum_engine.py"],
            fix_priority=1
        )
        
        fix_attempt = self.fix_system.apply_fix(diagnosis)
        
        assert fix_attempt.success is True
        assert "precision" in fix_attempt.fix_applied.lower()
        assert len(fix_attempt.notes) > 0
    
    def test_apply_fix_parameter_configuration(self):
        """Test applying parameter configuration fixes."""
        diagnosis = ErrorDiagnosis(
            error_type=ErrorType.PARAMETER_CONFIGURATION,
            test_name="parameter_test",
            description="Parameter issues",
            root_cause="Invalid parameters",
            suggested_fix="Optimize parameters",
            confidence=0.7,
            affected_files=["config.py"],
            fix_priority=2
        )
        
        fix_attempt = self.fix_system.apply_fix(diagnosis)
        
        assert fix_attempt.success is True
        assert "parameter" in fix_attempt.fix_applied.lower()
    
    def test_apply_fix_unsupported_type(self):
        """Test applying fix for unsupported error type."""
        diagnosis = ErrorDiagnosis(
            error_type=ErrorType.UNKNOWN,
            test_name="unknown_test",
            description="Unknown error",
            root_cause="Unknown cause",
            suggested_fix="Manual investigation required",
            confidence=0.1,
            affected_files=[],
            fix_priority=5
        )
        
        fix_attempt = self.fix_system.apply_fix(diagnosis)
        
        assert fix_attempt.success is False
        assert "no automatic fix available" in fix_attempt.notes.lower()
    
    def test_fix_attempt_timestamp(self):
        """Test that fix attempts have proper timestamps."""
        diagnosis = ErrorDiagnosis(
            error_type=ErrorType.TOLERANCE_THRESHOLD,
            test_name="test",
            description="Test",
            root_cause="Test",
            suggested_fix="Test",
            confidence=0.5,
            affected_files=[],
            fix_priority=1
        )
        
        before_time = time.time()
        fix_attempt = self.fix_system.apply_fix(diagnosis)
        after_time = time.time()
        
        # Parse timestamp and verify it's within reasonable range
        timestamp_str = fix_attempt.timestamp
        assert len(timestamp_str) > 0
        # Basic format check (YYYY-MM-DD HH:MM:SS)
        assert len(timestamp_str.split()) == 2
        assert len(timestamp_str.split()[0].split('-')) == 3
        assert len(timestamp_str.split()[1].split(':')) == 3


class TestChangelogUpdater:
    """Test changelog update functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.changelog_path = Path(self.temp_dir) / "CHANGELOG.md"
        self.changelog_updater = ChangelogUpdater(str(self.changelog_path))
    
    def test_initialization(self):
        """Test ChangelogUpdater initialization."""
        assert self.changelog_updater is not None
        assert self.changelog_updater.changelog_path == self.changelog_path
    
    def test_update_changelog_new_file(self):
        """Test updating changelog when file doesn't exist."""
        # Create mock debugging session
        session = DebuggingSession(
            session_id="test_session",
            start_time="2025-01-01 12:00:00",
            end_time="2025-01-01 12:30:00",
            initial_accuracy=95.0,
            final_accuracy=98.5,
            total_fixes_attempted=2,
            successful_fixes=2,
            error_diagnoses=[],
            fix_attempts=[
                FixAttempt(
                    diagnosis=ErrorDiagnosis(
                        error_type=ErrorType.TOLERANCE_THRESHOLD,
                        test_name="test1",
                        description="Test error 1",
                        root_cause="Test cause 1",
                        suggested_fix="Test fix 1",
                        confidence=0.9,
                        affected_files=[],
                        fix_priority=1
                    ),
                    fix_applied="Applied test fix 1",
                    timestamp="2025-01-01 12:15:00",
                    success=True,
                    accuracy_before=95.0,
                    accuracy_after=97.0,
                    notes="Test notes 1"
                )
            ],
            changelog_entries=[],
            success=True
        )
        
        result = self.changelog_updater.update_changelog(session)
        
        assert result is True
        assert self.changelog_path.exists()
        
        # Read and verify content
        content = self.changelog_path.read_text()
        assert "# CHANGELOG" in content
        assert "test_session" in content
        assert "95.0%" in content and "98.5%" in content  # Check for accuracy values
        assert "Applied test fix 1" in content
    
    def test_update_changelog_existing_file(self):
        """Test updating changelog when file already exists."""
        # Create existing changelog
        existing_content = """# CHANGELOG

## Version 1.1 - Previous Release

Some previous content.
"""
        self.changelog_path.write_text(existing_content)
        
        # Create mock debugging session
        session = DebuggingSession(
            session_id="new_session",
            start_time="2025-01-02 10:00:00",
            end_time="2025-01-02 10:15:00",
            initial_accuracy=96.0,
            final_accuracy=99.0,
            total_fixes_attempted=1,
            successful_fixes=1,
            error_diagnoses=[],
            fix_attempts=[],
            changelog_entries=[],
            success=True
        )
        
        result = self.changelog_updater.update_changelog(session)
        
        assert result is True
        
        # Read and verify content
        content = self.changelog_path.read_text()
        assert "# CHANGELOG" in content
        assert "new_session" in content
        assert "96.0%" in content and "99.0%" in content  # Check for accuracy values
        assert "Version 1.1 - Previous Release" in content  # Existing content preserved
    
    def test_generate_changelog_entry(self):
        """Test changelog entry generation."""
        session = DebuggingSession(
            session_id="test_entry",
            start_time="2025-01-01 15:00:00",
            end_time="2025-01-01 15:20:00",
            initial_accuracy=90.0,
            final_accuracy=98.0,
            total_fixes_attempted=3,
            successful_fixes=2,
            error_diagnoses=[],
            fix_attempts=[
                FixAttempt(
                    diagnosis=ErrorDiagnosis(
                        error_type=ErrorType.NUMERICAL_PRECISION,
                        test_name="precision_test",
                        description="Precision error",
                        root_cause="Low precision",
                        suggested_fix="Increase precision",
                        confidence=0.8,
                        affected_files=[],
                        fix_priority=1
                    ),
                    fix_applied="Increased numerical precision",
                    timestamp="2025-01-01 15:10:00",
                    success=True,
                    accuracy_before=90.0,
                    accuracy_after=95.0,
                    notes="Enhanced matrix operations"
                )
            ],
            changelog_entries=[],
            success=True
        )
        
        entry = self.changelog_updater._generate_changelog_entry(session)
        
        assert "Version 1.2" in entry
        assert "test_entry" in entry
        assert "90.0%" in entry and "98.0%" in entry
        assert "2/3" in entry
        assert "precision_test" in entry
        assert "Increased numerical precision" in entry
        assert "ACHIEVED" in entry


class TestDebuggingLoop:
    """Test complete debugging loop functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock validator
        self.mock_validator = Mock(spec=QBESValidator)
        self.debugging_loop = DebuggingLoop(validator=self.mock_validator)
    
    def test_initialization(self):
        """Test DebuggingLoop initialization."""
        assert self.debugging_loop is not None
        assert self.debugging_loop.validator == self.mock_validator
        assert hasattr(self.debugging_loop, 'error_detector')
        assert hasattr(self.debugging_loop, 'fix_system')
        assert hasattr(self.debugging_loop, 'changelog_updater')
    
    def test_initialization_with_default_validator(self):
        """Test DebuggingLoop initialization with default validator."""
        loop = DebuggingLoop()
        assert loop.validator is not None
        assert isinstance(loop.validator, QBESValidator)
    
    @patch('time.strftime')
    def test_execute_debugging_loop_success(self, mock_strftime):
        """Test successful debugging loop execution."""
        mock_strftime.return_value = "2025-01-01 12:00:00"
        
        # Mock initial validation (low accuracy)
        initial_summary = Mock(spec=ValidationSummary)
        initial_summary.final_accuracy = 95.0
        initial_summary.failed_tests = ["test1"]
        initial_summary.retry_history = [{"attempt": 1}]
        
        # Mock improved validation (high accuracy)
        improved_summary = Mock(spec=ValidationSummary)
        improved_summary.final_accuracy = 98.5
        improved_summary.failed_tests = []
        improved_summary.retry_history = [{"attempt": 1}]
        
        # Configure mock validator
        self.mock_validator.validate_against_benchmarks.side_effect = [
            initial_summary, improved_summary
        ]
        
        # Execute debugging loop
        session = self.debugging_loop.execute_debugging_loop(
            max_iterations=2,
            target_accuracy=98.0,
            suite="standard"
        )
        
        # Verify session results
        assert isinstance(session, DebuggingSession)
        assert session.initial_accuracy == 95.0
        assert session.final_accuracy == 98.5
        assert session.success is True
        assert session.total_fixes_attempted > 0
        assert len(session.error_diagnoses) > 0
    
    @patch('time.strftime')
    def test_execute_debugging_loop_no_improvement(self, mock_strftime):
        """Test debugging loop when no improvement is achieved."""
        mock_strftime.return_value = "2025-01-01 12:00:00"
        
        # Mock validation that doesn't improve
        summary = Mock(spec=ValidationSummary)
        summary.final_accuracy = 95.0
        summary.failed_tests = ["test1"]
        summary.retry_history = [{"attempt": 1}]
        
        self.mock_validator.validate_against_benchmarks.return_value = summary
        
        # Execute debugging loop
        session = self.debugging_loop.execute_debugging_loop(
            max_iterations=2,
            target_accuracy=98.0,
            suite="standard"
        )
        
        # Verify session results
        assert session.success is False
        assert session.final_accuracy < 98.0
    
    @patch('time.strftime')
    def test_execute_debugging_loop_no_errors_detected(self, mock_strftime):
        """Test debugging loop when no fixable errors are detected."""
        mock_strftime.return_value = "2025-01-01 12:00:00"
        
        # Mock validation with high accuracy and no failed tests
        summary = Mock(spec=ValidationSummary)
        summary.final_accuracy = 99.0  # Above target, should not trigger accuracy diagnosis
        summary.failed_tests = []  # No failed tests to diagnose
        summary.retry_history = [{"attempt": 1}]  # Single attempt, no retry pattern
        
        self.mock_validator.validate_against_benchmarks.return_value = summary
        
        # Execute debugging loop
        session = self.debugging_loop.execute_debugging_loop(
            max_iterations=2,
            target_accuracy=98.0,
            suite="standard"
        )
        
        # Should stop early due to no detectable errors and target already achieved
        assert session.success is True  # Target accuracy achieved
        assert session.total_fixes_attempted == 0
    
    def test_log_debugging_summary(self):
        """Test debugging summary logging."""
        # Create mock session
        session = DebuggingSession(
            session_id="test_log",
            start_time="2025-01-01 12:00:00",
            end_time="2025-01-01 12:30:00",
            initial_accuracy=95.0,
            final_accuracy=98.5,
            total_fixes_attempted=2,
            successful_fixes=2,
            error_diagnoses=[],
            fix_attempts=[],
            changelog_entries=[],
            success=True
        )
        
        self.debugging_loop.current_session = session
        
        # Should not raise any exceptions
        self.debugging_loop._log_debugging_summary()
    
    def test_log_debugging_summary_no_session(self):
        """Test debugging summary logging with no current session."""
        self.debugging_loop.current_session = None
        
        # Should not raise any exceptions
        self.debugging_loop._log_debugging_summary()


class TestIntegration:
    """Integration tests for the complete debugging system."""
    
    def test_error_detection_to_fix_application(self):
        """Test integration from error detection to fix application."""
        # Create error detector and fix system
        error_detector = ErrorDetectionSystem()
        fix_system = AutomaticFixSystem()
        
        # Create mock validation summary with errors
        validation_summary = Mock(spec=ValidationSummary)
        validation_summary.failed_tests = ["tolerance_test"]
        validation_summary.final_accuracy = 95.0
        validation_summary.retry_history = [{"attempt": 1}]
        
        # Detect errors
        diagnoses = error_detector.detect_errors(validation_summary)
        assert len(diagnoses) > 0
        
        # Apply fixes
        for diagnosis in diagnoses:
            fix_attempt = fix_system.apply_fix(diagnosis)
            assert isinstance(fix_attempt, FixAttempt)
            if diagnosis.error_type in [ErrorType.TOLERANCE_THRESHOLD, 
                                      ErrorType.NUMERICAL_PRECISION, 
                                      ErrorType.PARAMETER_CONFIGURATION]:
                assert fix_attempt.success is True
    
    def test_complete_workflow_simulation(self):
        """Test complete debugging workflow simulation."""
        # This test simulates the complete workflow without actual file modifications
        
        # Create debugging loop with mock validator
        mock_validator = Mock(spec=QBESValidator)
        debugging_loop = DebuggingLoop(validator=mock_validator)
        
        # Mock validation results
        low_accuracy_summary = Mock(spec=ValidationSummary)
        low_accuracy_summary.final_accuracy = 95.0
        low_accuracy_summary.failed_tests = ["precision_test"]
        low_accuracy_summary.retry_history = [{"attempt": 1}]
        
        high_accuracy_summary = Mock(spec=ValidationSummary)
        high_accuracy_summary.final_accuracy = 98.5
        high_accuracy_summary.failed_tests = []
        high_accuracy_summary.retry_history = [{"attempt": 1}]
        
        mock_validator.validate_against_benchmarks.side_effect = [
            low_accuracy_summary, high_accuracy_summary
        ]
        
        # Execute debugging loop
        with patch('time.strftime', return_value="2025-01-01 12:00:00"):
            session = debugging_loop.execute_debugging_loop(
                max_iterations=2,
                target_accuracy=98.0,
                suite="test"
            )
        
        # Verify complete workflow
        assert session.success is True
        assert session.initial_accuracy == 95.0
        assert session.final_accuracy == 98.5
        assert session.total_fixes_attempted > 0
        assert len(session.error_diagnoses) > 0
        assert len(session.fix_attempts) > 0


if __name__ == "__main__":
    pytest.main([__file__])