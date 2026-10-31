@echo off
echo ========================================================
echo Running Spinal Cord MRI Backend Test Suite...
echo ========================================================
"%~dp0app\app\backend\venv\Scripts\pytest.exe" "%~dp0app\app\backend\tests\test_mri_backend.py" -v
pause
