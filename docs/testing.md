# Testing Guidelines

## Unit Tests

Unit tests should cover the following components:

1. **Image Preprocessing**
   - Test image enhancement functions
   - Verify correct normalization
   - Check output dimensions

2. **Dental Feature Detection**
   - Validate description generation
   - Test dental terminology enhancement
   - Check for correct condition identification

3. **Report Generation**
   - Verify report structure
   - Test content generation
   - Check for required sections

## Integration Tests

1. **End-to-End Pipeline**
   - Test complete analysis workflow
   - Verify data flow between components
   - Check error handling

2. **Web Interface**
   - Test image upload functionality
   - Verify report display
   - Check error messages

## Test Data

For testing, use the provided sample X-ray image:
- `temp_dental_xray.jpg`

Additional test images should cover:
- Normal dental X-rays
- X-rays with common pathologies
- Different image formats and sizes

## Running Tests

To run tests:
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_preprocessing.py

# Run with coverage
python -m pytest --cov=src tests/
```

## Test Environment

Ensure the following are installed for testing:
- pytest
- pytest-cov (for coverage reports)
- mock (for mocking external services)

Install with:
```bash
pip install pytest pytest-cov mock
```