@echo off
REM ============================================================================
REM SARA TTS Auto-Setup and Runner Script
REM This script automatically sets up environment and downloads necessary files
REM ============================================================================

echo.
echo ============================================================================
echo                     SARA TTS Auto-Setup and Runner
echo ============================================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Set variables
set VENV_PATH=venv
set PYTHON_SCRIPT=sara_tts.py
set CONFIG_FILE=config.json
set REQUIREMENTS_FILE=requirements.txt

echo Current directory: %CD%
echo.

REM ============================================================================
REM Check Python installation
REM ============================================================================
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    goto :keep_open
)

python --version
echo Python found successfully.
echo.

REM ============================================================================
REM Create virtual environment if not exists
REM ============================================================================
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv %VENV_PATH%
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        echo.
        goto :keep_open
    )
    echo Virtual environment created successfully.
    echo.
) else (
    echo Virtual environment already exists.
    echo.
)

REM ============================================================================
REM Activate virtual environment
REM ============================================================================
echo Activating virtual environment...
call .\%VENV_PATH%\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    echo.
    goto :keep_open
)

echo Virtual environment activated successfully.
echo.

REM ============================================================================
REM Create requirements.txt if not exists
REM ============================================================================
if not exist "%REQUIREMENTS_FILE%" (
    echo Creating requirements.txt...
    (
        echo torch
        echo transformers
        echo soundfile
        echo librosa
        echo numpy
    ) > %REQUIREMENTS_FILE%
    echo Requirements file created.
    echo.
)

REM ============================================================================
REM Install/upgrade packages
REM ============================================================================
echo Checking and installing required packages...
echo This may take a few minutes on first run...
echo.

pip install --upgrade pip
pip install -r %REQUIREMENTS_FILE%

if errorlevel 1 (
    echo ERROR: Failed to install some packages!
    echo Trying individual installation...
    echo.
    
    echo Installing torch...
    pip install torch
    
    echo Installing transformers...
    pip install transformers
    
    echo Installing soundfile...
    pip install soundfile
    
    echo Installing librosa...
    pip install librosa
    
    echo Installing numpy...
    pip install numpy
)

echo Package installation completed.
echo.

REM ============================================================================
REM Create Python script if not exists
REM ============================================================================
if not exist "%PYTHON_SCRIPT%" (
    echo Python script not found. Creating default script...
    echo.
    
    powershell -Command "& {
        $url = 'https://raw.githubusercontent.com/user/repo/main/sara_tts.py'
        try {
            Invoke-WebRequest -Uri $url -OutFile 'sara_tts.py' -ErrorAction Stop
            Write-Host 'Script downloaded successfully from GitHub.'
        } catch {
            Write-Host 'Could not download from GitHub. Creating basic template...'
            $template = @'
import torch
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
import os
import numpy as np
import json

# Basic SARA TTS implementation
# Replace this with your full implementation

def main():
    print(\"SARA TTS Script Template\")
    print(\"Please replace this with your full implementation.\")
    print(\"Check README.md for complete code.\")

if __name__ == \"__main__\":
    main()
'@
            $template | Out-File -FilePath 'sara_tts.py' -Encoding UTF8
            Write-Host 'Template script created. Please add your full implementation.'
        }
    }"
    
    echo.
)

REM ============================================================================
REM Create sample configuration if not exists
REM ============================================================================
if not exist "%CONFIG_FILE%" (
    echo Creating sample configuration file...
    (
        echo [
        echo     {
        echo         "text": "Salam, necəsən? Bu mənim test mətnimdir.",
        echo         "length_scale": 1.0,
        echo         "pitch_shift": 0.0,
        echo         "use_pipeline": false,
        echo         "output_suffix": "_normal"
        echo     },
        echo     {
        echo         "text": "Bu mətn sürətli şəkildə oxunacaq.",
        echo         "length_scale": 0.7,
        echo         "pitch_shift": 0.0,
        echo         "use_pipeline": false,
        echo         "output_suffix": "_fast"
        echo     },
        echo     {
        echo         "text": "Bu mətn yavaş şəkildə oxunacaq.",
        echo         "length_scale": 1.5,
        echo         "pitch_shift": 4.0,
        echo         "use_pipeline": false,
        echo         "output_suffix": "_slow_high"
        echo     }
        echo ]
    ) > %CONFIG_FILE%
    echo Sample configuration created: %CONFIG_FILE%
    echo.
)

REM ============================================================================
REM Create output directory
REM ============================================================================
if not exist "output_audio" (
    mkdir output_audio
    echo Created output_audio directory.
    echo.
)

REM ============================================================================
REM Display setup summary
REM ============================================================================
echo ============================================================================
echo                           SETUP SUMMARY
echo ============================================================================
echo Virtual Environment: %VENV_PATH% - OK
echo Python Script: %PYTHON_SCRIPT% - %~z1 bytes
echo Configuration: %CONFIG_FILE% - OK
echo Output Directory: output_audio - OK
echo.

REM Check if all files exist
set SETUP_COMPLETE=1

if not exist "%PYTHON_SCRIPT%" (
    echo WARNING: Python script is missing or incomplete!
    set SETUP_COMPLETE=0
)

if not exist "%CONFIG_FILE%" (
    echo WARNING: Configuration file is missing!
    set SETUP_COMPLETE=0
)

if %SETUP_COMPLETE%==0 (
    echo.
    echo SETUP INCOMPLETE! Please check the warnings above.
    echo.
    goto :keep_open
)

REM ============================================================================
REM Run the script
REM ============================================================================
echo ============================================================================
echo                        Starting SARA TTS Script
echo ============================================================================
echo.

REM Test import first
echo Testing Python environment...
python -c "import torch, transformers, soundfile; print('All required packages imported successfully')"

if errorlevel 1 (
    echo ERROR: Package import failed! Please check the installation.
    echo.
    goto :keep_open
)

echo.
echo Running SARA TTS script...
echo.

REM Run the main script
python %PYTHON_SCRIPT%

REM Check exit code
if errorlevel 1 (
    echo.
    echo ============================================================================
    echo                           SCRIPT FAILED
    echo ============================================================================
    echo The script encountered an error. Check the output above for details.
    echo.
    echo Common solutions:
    echo - Check your config.json format
    echo - Ensure internet connection for model download
    echo - Verify GPU drivers if using CUDA
    echo.
) else (
    echo.
    echo ============================================================================
    echo                        SCRIPT COMPLETED SUCCESSFULLY
    echo ============================================================================
    echo Generated audio files are in the 'output_audio' folder.
    echo.
    
    REM Show output files
    if exist "output_audio\*.wav" (
        echo Generated files:
        dir /b "output_audio\*.wav"
        echo.
    )
)

REM ============================================================================
REM Cleanup and exit
REM ============================================================================
:keep_open
echo ============================================================================
echo                              FINISHED
echo ============================================================================
echo.
echo To run again: Double-click this batch file
echo To modify settings: Edit config.json
echo For help: Check README.md
echo.
echo Press any key to close this window...
pause >nul

REM Deactivate virtual environment
call deactivate >nul 2>&1