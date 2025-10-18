"""
Pytest configuration and shared fixtures for LNN examples testing

Author: Gianluca Mazza
"""

import pytest
import sys
from pathlib import Path

# Add examples directory to path
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))


@pytest.fixture
def tolerance():
    """Default tolerance for floating point comparisons"""
    return 0.1


@pytest.fixture
def high_tolerance():
    """Higher tolerance for less precise comparisons"""
    return 0.2
