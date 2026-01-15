"""
Pytest configuration for test timeouts.

This conftest.py automatically sets timeouts for test files based on their
location and characteristics. Individual tests can override these defaults
using @pytest.mark.timeout().
"""

import pytest
import os


def pytest_collection_modifyitems(config, items):
    """
    Automatically set timeouts for test files based on their path.
    
    This function runs after test collection but before execution,
    allowing us to set different default timeouts for different test files.
    """
    for item in items:
        # Only set timeout if no explicit timeout marker is present
        if item.get_closest_marker("timeout"):
            continue
        
        file_path = str(item.fspath)
        
        # Set timeouts based on test file location
        if "/comprehensive/" in file_path:
            # Comprehensive tests may take longer
            item.add_marker(pytest.mark.timeout(600))  # 10 minutes
        elif "/unit/" in file_path:
            # Unit tests should be fast
            item.add_marker(pytest.mark.timeout(120))  # 2 minutes
        elif "test_auto_active" in file_path:
            # Auto-active tests may take longer
            item.add_marker(pytest.mark.timeout(600))  # 10 minutes
        else:
            # Default timeout for other tests (can be overridden by pytest.ini)
            # This will use the global timeout from pytest.ini (300 seconds)
            pass
