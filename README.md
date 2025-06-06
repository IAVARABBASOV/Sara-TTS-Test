# SARA TTS - Azerbaijani Text-to-Speech with Speed & Pitch Control

A Python implementation for converting Azerbaijani text to speech using the BHOSAI/SARA_TTS model with advanced audio manipulation capabilities.

## Features

- **Azerbaijani TTS**: High-quality text-to-speech conversion using BHOSAI/SARA_TTS VITS model
- **Speed Control**: Adjustable speech rate via `length_scale` parameter
- **Pitch Control**: Voice pitch adjustment in semitones (-24 to +24 range)
- **JSON Configuration**: Batch processing support with configuration files
- **Multiple Outputs**: Process multiple configurations in a single run
- **GPU Acceleration**: CUDA support for faster processing
- **Flexible Input**: Supports both single and multiple configuration formats

## Requirements

### Core Dependencies
```bash
pip install torch transformers soundfile
```

### Optional (for advanced audio processing)
```bash
pip install librosa
```

## Installation

### Automatic Installation (Windows)
1. Download all project files including `setup.bat`
2. Double-click `setup.bat`
3. The script will automatically:
   - Verify Python installation
   - Create virtual environment
   - Install latest versions of: `torch`, `transformers`, `soundfile`, `librosa`, `numpy`
   - Create sample configuration files
   - Run the TTS script

### Manual Installation (All Platforms)
1. Clone or download the script
2. Create virtual environment:
```bash
python -m venv venv
```
3. Activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```
4. Install dependencies:
```bash
pip install torch transformers soundfile librosa numpy
```
5. Run the script:
```bash
python sara_tts.py
```

## Usage

### Configuration File Format

Create a `config.json` file in one of these formats:

#### Single Configuration
```json
{
    "text": "Salam, necəsən? Bu mənim test mətnimdir.",
    "length_scale": 1.0,
    "pitch_shift": 0.0,
    "use_pipeline": false,
    "output_suffix": "_custom"
}
```

#### Multiple Configurations
```json
[
    {
        "text": "Salam, necəsən? Bu mənim test mətnimdir.",
        "length_scale": 1.0,
        "pitch_shift": -2.0,
        "use_pipeline": false,
        "output_suffix": "_normal"
    },
    {
        "text": "Bu mətn sürətli şəkildə oxunacaq.",
        "length_scale": 0.7,
        "pitch_shift": 0.0,
        "use_pipeline": false,
        "output_suffix": "_fast"
    },
    {
        "text": "Bu mətn yavaş şəkildə oxunacaq.",
        "length_scale": 1.5,
        "pitch_shift": 4.0,
        "use_pipeline": false,
        "output_suffix": "_slow_high"
    }
]
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | **Required** | Azerbaijani text to convert to speech |
| `length_scale` | float | 1.0 | Speed control (1.0=normal, <1.0=faster, >1.0=slower) |
| `pitch_shift` | float | 0.0 | Pitch adjustment in semitones (-24 to +24) |
| `use_pipeline` | boolean | false | Use pipeline approach (limited pitch control) |
| `output_suffix` | string | "" | Custom suffix for output filename |

### Speed Examples
- `0.5` - 2x faster
- `0.8` - 25% faster  
- `1.0` - Normal speed
- `1.3` - 30% slower
- `2.0` - 2x slower

### Pitch Examples
- `-12` - One octave lower
- `-3` - Minor third down
- `0` - Normal pitch
- `3` - Minor third up
- `12` - One octave higher

## Running the Script

### Option 1: Using Batch Files (Windows - Recommended)

#### Auto-Setup and Run (setup.bat)
For complete automation including environment setup:
```cmd
setup.bat
```

**Features:**
- ✅ Checks Python installation
- ✅ Creates virtual environment automatically
- ✅ Installs all required packages (latest versions)
- ✅ Creates sample configuration files
- ✅ Creates output directory
- ✅ Runs the TTS script
- ✅ Shows generated files

#### Simple Runner (run.bat)
For subsequent runs when environment is already set up:
```cmd
run.bat
```

**Features:**
- ✅ Activates existing virtual environment
- ✅ Runs the TTS script
- ✅ Shows execution status

### Option 2: Manual Python Execution

1. **With existing config.json**:
```bash
python sara_tts.py
```

2. **First run** (creates sample configurations):
```bash
python sara_tts.py
# Will create: config.json, config_fast.json, config_slow.json, config_high.json, config_low.json
```

## Output

Audio files are saved in the `output_audio/` directory with descriptive filenames:
```
sara_output_speed_1_0_pitch_0_0_normal.wav
sara_output_speed_0_7_pitch_0_0_fast.wav
sara_output_speed_1_5_pitch_4_0_slow_high.wav
```

## File Structure

```
project/
├── sara_tts.py          # Main TTS script
├── setup.bat            # Auto-setup and runner (Windows)
├── run.bat              # Simple runner (Windows)
├── config.json          # Your configuration file
├── requirements.txt     # Python dependencies (auto-created)
├── venv/               # Virtual environment (auto-created)
├── output_audio/        # Generated audio files
│   ├── sara_output_*.wav
│   └── ...
└── README.md           # This file
```

## Quick Start Guide

### For Windows Users
1. **Download all files** to a folder
2. **Double-click `setup.bat`** - This handles everything automatically:
   - Checks Python installation
   - Creates virtual environment
   - Installs required packages
   - Creates sample configurations
   - Runs the script
3. **Edit `config.json`** to customize your text and settings
4. **Double-click `run.bat`** for subsequent runs

### For Linux/Mac Users
1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install packages: `pip install torch transformers soundfile librosa`
4. Run script: `python sara_tts.py`

## GPU Support

The script automatically detects and uses CUDA if available:
- **GPU**: Faster processing with CUDA-enabled PyTorch
- **CPU**: Fallback mode (slower but functional)

Check GPU status in output:
```
Using GPU: NVIDIA GeForce RTX 4090
Device set to use cuda
```

## Troubleshooting

### Batch File Issues (Windows)

**Python not found**:
```
ERROR: Python is not installed or not in PATH!
```
**Solution**: Install Python from https://python.org and check "Add Python to PATH"

**Virtual environment creation failed**:
```
ERROR: Failed to create virtual environment!
```
**Solution**: Run as Administrator or check disk space

**Package installation failed**:
```
ERROR: Failed to install some packages!
```
**Solution**: Check internet connection, try running as Administrator

**Script not found**:
```
WARNING: Python script is missing or incomplete!
```
**Solution**: Ensure `sara_tts.py` is in the same folder as batch files

### Common Script Issues

**JSON Format Error**:
```
Error: Invalid JSON format in 'config.json'
JSON Error: Extra data: line 7 column 2 (char 168)
```
**Solution**: Wrap multiple configurations in square brackets `[]`

**Model Loading Error**:
```
Error loading model or tokenizer: ...
```
**Solution**: Check internet connection for model download

**Audio Processing Error**:
```
Speed/Pitch adjustment failed: ...
```
**Solution**: Install librosa: `pip install librosa`

**Empty Audio Output**:
```
WARNING: audio_array is empty!
```
**Solution**: Check text content and model compatibility

### Dependencies Issues

**Missing librosa**:
- Speed/pitch adjustments will be skipped
- Install with: `pip install librosa`

**CUDA not detected**:
- Script will use CPU (slower)
- Install PyTorch with CUDA support

## Technical Details

### Model Information
- **Model**: BHOSAI/SARA_TTS (VITS-based)
- **Language**: Azerbaijani
- **Architecture**: Variational Inference with adversarial learning for end-to-end Text-to-Speech
- **Sampling Rate**: 22050 Hz (model default)

### Audio Processing
- **Speed Control**: Uses librosa time-stretching without pitch change
- **Pitch Control**: Uses librosa pitch-shifting without tempo change
- **Output Format**: WAV files with original sampling rate

### Performance
- **GPU**: ~2-5 seconds per sentence
- **CPU**: ~10-30 seconds per sentence
- **Memory**: ~2-4GB VRAM (GPU mode)

## Advanced Usage

### Custom Output Directory
Modify `output_folder` parameter in the script:
```python
output_path = test_sara_tts_direct(
    text, 
    output_folder="my_custom_folder"
)
```

### Batch Processing Script
```python
# Process multiple files
config_files = ["config1.json", "config2.json", "config3.json"]
for config_file in config_files:
    process_json_config(config_file)
```

## License

This project uses the BHOSAI/SARA_TTS model. Check the model's license terms on Hugging Face.

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

## Support

For issues related to:
- **Batch Files**: Ensure Python is installed and added to PATH, run as Administrator if needed
- **Model**: Check [BHOSAI/SARA_TTS on Hugging Face](https://huggingface.co/BHOSAI/SARA_TTS)
- **Dependencies**: Refer to respective package documentation (torch, transformers, librosa)
- **Script**: Create an issue with error details and configuration

### Getting Help
1. **Check this README** for common solutions
2. **Run `setup.bat`** to ensure proper environment setup
3. **Verify `config.json`** format is correct
4. **Check console output** for specific error messages
5. **Try manual installation** if batch files fail