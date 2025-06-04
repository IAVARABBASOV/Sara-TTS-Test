@echo off
REM ============================================================================
REM SARA TTS Runner Script
REM This script activates the virtual environment and runs the SARA TTS script
REM ============================================================================

echo.
echo ============================================================================
echo                           SARA TTS Runner
echo ============================================================================
echo.

REM Change to script directory (in case bat file is run from different location)
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please create a virtual environment first:
    echo.
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install torch transformers soundfile librosa
    echo.
    goto :keep_open
)

REM Check if Python script exists
if not exist "test_sara_tts.py" (
    echo ERROR: Python script 'test_sara_tts.py' not found!
    echo Please make sure the script is in the same directory as this batch file.
    echo.
    goto :keep_open
)

echo Activating virtual environment...
call .\venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    echo.
    goto :keep_open
)

echo Virtual environment activated successfully.
echo.

echo Current directory: %CD%
echo Python version:
python --version
echo.

echo ============================================================================
echo                        Starting SARA TTS Script
echo ============================================================================
echo.

REM Run the Python script
python test_sara_tts.py

REM Check if script ran successfully
if errorlevel 1 (
    echo.
    echo ============================================================================
    echo                           SCRIPT FAILED
    echo ============================================================================
    echo The script encountered an error. Please check the output above.
    echo.
) else (
    echo.
    echo ============================================================================
    echo                        SCRIPT COMPLETED
    echo ============================================================================
    echo The script has finished running successfully.
    echo Check the 'output_audio' folder for generated audio files.
    echo.
)

:keep_open
echo.
echo Press any key to close this window...
pause >nul