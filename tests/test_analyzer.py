import unittest
import numpy as np
from PIL import Image
import sys
import os

# Add src to path to import the DentalXRayAnalyzer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import DentalXRayAnalyzer

class TestDentalXRayAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = DentalXRayAnalyzer()
        
    def test_preprocess_xray(self):
        """Test the preprocess_xray method."""
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='white')
        
        # Save the test image
        test_image_path = 'test_image.jpg'
        test_image.save(test_image_path)
        
        # Test preprocessing
        enhanced_image = self.analyzer.preprocess_xray(test_image_path)
        
        # Check that we get a PIL Image back
        self.assertIsInstance(enhanced_image, Image.Image)
        
        # Clean up
        os.remove(test_image_path)
        
    def test_enhance_dental_description(self):
        """Test the enhance_dental_description method."""
        # Test with a description containing dental terms
        description = "The image shows dark spots on teeth"
        enhanced = self.analyzer.enhance_dental_description(description)
        
        # Check that the enhanced description contains expected elements
        self.assertIn("COMPREHENSIVE DENTAL X-RAY ANALYSIS", enhanced)
        self.assertIn("Dental Conditions Detected", enhanced)
        
        # Test with a neutral description
        description = "The image shows a clear view of teeth"
        enhanced = self.analyzer.enhance_dental_description(description)
        
        # Check that it handles neutral descriptions
        self.assertIn("COMPREHENSIVE DENTAL X-RAY ANALYSIS", enhanced)
        self.assertIn("No specific dental pathologies identified", enhanced)

if __name__ == '__main__':
    unittest.main()