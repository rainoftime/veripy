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
    'tests/unit/test_extended_features.py': 30,
    'tests/unit/test_extended_python_features.py': 30,
    'tests/unit/test_decorators.py': 30,
    'tests/unit/test_variable_arguments.py': 30,
    'tests/unit/test_fstrings.py': 30,
    'tests/unit/test_dataclasses.py': 30,
    'tests/unit/test_type_system.py': 30,
    'tests/unit/test_iteration.py': 30,
    'tests/unit/test_builtins.py': 30,
    'tests/unit/test_syntax_exports.py': 30,
    
    # Split comprehensive tests - now in unit directory
    'tests/unit/test_control_flow.py': 30,
    'tests/unit/test_data_types.py': 0,
    'tests/unit/test_expressions.py': 60,
    'tests/unit/test_functions.py': 60,
    'tests/unit/test_statements.py': 60,
    'tests/unit/test_edge_cases.py': 60,
    'tests/unit/test_loop_invariants.py': 90,
    'tests/unit/test_quantifiers_comprehensive.py': 90,
    'tests/unit/test_termination.py': 90,
    'tests/unit/test_frame_conditions.py': 90,
    'tests/unit/test_automatic_inference.py': 90,
    'tests/unit/test_complex_verification.py': 120,
    'tests/unit/test_set_operations.py': 60,
    'tests/unit/test_dictionary_operations.py': 60,
    'tests/unit/test_string_operations.py': 60,
    'tests/unit/test_comprehensions.py': 60,
    'tests/unit/test_field_access.py': 60,
    'tests/unit/test_method_calls.py': 60,
    'tests/unit/test_extended_combinations.py': 60,
    'tests/unit/test_sorting_algorithms.py': 120,
    'tests/unit/test_search_algorithms.py': 120,
    'tests/unit/test_prefix_sum.py': 90,
    'tests/unit/test_two_pointers.py': 90,
    'tests/unit/test_bit_manipulation.py': 90,
    'tests/unit/test_mathematical_properties.py': 90,
    'tests/unit/test_array_manipulation.py': 90,
    'tests/unit/test_string_algorithms.py': 90,
    'tests/unit/test_advanced_quantifiers.py': 90,
    
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
            else:
                timeout = TEST_FILE_TIMEOUTS.get('default', 60)
        
        # Set timeout using pytest-timeout marker
        # Remove any existing timeout markers first
        item.own_markers = [m for m in item.own_markers if m.name != 'timeout']
        # Use 'signal' method instead of 'thread' - works better with blocking C code (like Z3 solver)
        # Signal method can interrupt blocking operations, thread method cannot
        item.add_marker(pytest.mark.timeout(timeout, method='signal'))
