"""
Pytest configuration for veripy tests.

Sets timeouts per test file to prevent hanging tests.
"""
import pytest
import os

# Timeout configuration per test file (in seconds)
TEST_FILE_TIMEOUTS = {
    # Unit tests - quick, should complete in under 30 seconds
    'tests/unit/test_quantifiers.py': 30,
    'tests/unit/test_loops.py': 30,
    'tests/unit/test_calls.py': 30,
    'tests/unit/test_refinement_types.py': 30,
    'tests/unit/test_arrays.py': 30,
    'tests/unit/test_structures.py': 30,
    'tests/unit/test_decreases.py': 30,
    'tests/unit/test_frames.py': 30,
    'tests/unit/test_counter.py': 30,
    'tests/unit/test_extended_features.py': 60,
    'tests/unit/test_extended_python_features.py': 60,
    'tests/unit/test_advanced_python_features.py': 60,
    
    # Comprehensive tests - may take longer
    'tests/comprehensive/test_auto_active_features.py': 120,
    'tests/comprehensive/test_dafny_verus_patterns.py': 180,
    'tests/comprehensive/test_extended_features.py': 120,
    'tests/comprehensive/test_python_features.py': 120,
    
    # Integration tests
    'tests/test_auto_active.py': 90,
    'tests/test_cases.py': 60,
    'tests/test_prototype.py': 60,
    
    # Default timeout for any other test file
    'default': 60,
}

def pytest_collection_modifyitems(config, items):
    """Set timeout markers based on test file paths."""
    for item in items:
        # Get the test file path relative to project root
        try:
            test_file = os.path.relpath(str(item.fspath))
        except (AttributeError, ValueError):
            # Fallback if path resolution fails
            test_file = str(item.fspath)
        
        # Normalize path separators
        test_file = test_file.replace('\\', '/')
        
        # Find matching timeout
        timeout = TEST_FILE_TIMEOUTS.get(test_file)
        if timeout is None:
            # Try to match by directory pattern
            if 'tests/unit/' in test_file:
                timeout = 30
            elif 'tests/comprehensive/' in test_file:
                timeout = 120
            else:
                timeout = TEST_FILE_TIMEOUTS.get('default', 60)
        
        # Set timeout using pytest-timeout marker
        # Remove any existing timeout markers first
        item.own_markers = [m for m in item.own_markers if m.name != 'timeout']
        item.add_marker(pytest.mark.timeout(timeout, method='thread'))
