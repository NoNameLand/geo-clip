"""
Basic test for GeoCLIP package structure and imports.

This test verifies that the main GeoCLIP components can be imported
and instantiated without errors.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that core GeoCLIP components can be imported."""
    from geoclip_og import GeoCLIP, LocationEncoder, ImageEncoder
    # If import fails, pytest will mark this test as failed automatically.
    print("✅ Core imports successful")


def test_model_instantiation():
    """Test that models can be instantiated."""
    from geoclip_og import GeoCLIP, LocationEncoder

    # Test LocationEncoder (doesn't require pre-trained weights)
    location_encoder = LocationEncoder()
    print("✅ LocationEncoder instantiated")

    # Note: GeoCLIP requires pre-trained weights, so we skip full instantiation
    print("✅ Model instantiation test passed")

def test_package_structure():
    """Test that package structure is correct."""
    import geoclip_og as geoclip

    # Check that main components are available
    required_components = ['GeoCLIP', 'LocationEncoder', 'ImageEncoder']

    for component in required_components:
        assert hasattr(geoclip, component), f"{component} missing from geoclip package"
        print(f"✅ {component} available in geoclip package")

## Tests are executed via pytest. No __main__ runner is needed.
