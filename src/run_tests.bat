@echo off
REM run_tests.bat - Windows test runner

echo ==================================================
echo   Smart Labeling System - Automated Test Suite
echo ==================================================
echo.

REM Check Python availability
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo Using: python
echo.

REM Run test suite
echo Starting test suite...
echo.

python test_system.py

REM Capture exit code
set EXIT_CODE=%ERRORLEVEL%

echo.
echo ==================================================

if %EXIT_CODE% EQU 0 (
    echo ALL TESTS PASSED - System ready!
) else if %EXIT_CODE% EQU 1 (
    echo PARTIAL PASS - Check warnings above
) else if %EXIT_CODE% EQU 2 (
    echo TESTS FAILED - Critical issues found
) else (
    echo ERROR - Test suite crashed
)

echo ==================================================
echo.
pause

exit /b %EXIT_CODE%