import torch
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
import os
import numpy as np
import json

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Librosa not available. Install with: pip install librosa")

def read_json_config(file_path):
    """
    Read configuration from JSON file. Supports both single objects and arrays.
    
    Args:
        file_path: Path to the JSON configuration file
        
    Returns:
        list: List of configuration dictionaries, or None if error occurs
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Handle both single object and array of objects
        if isinstance(data, dict):
            configs = [data]
        elif isinstance(data, list):
            configs = data
        else:
            print(f"Error: JSON must contain either an object or array of objects")
            return None
        
        processed_configs = []
        
        for i, config in enumerate(configs):
            if not isinstance(config, dict):
                print(f"Error: Configuration {i+1} is not a valid object")
                continue
                
            # Validate required fields
            if 'text' not in config:
                print(f"Error: 'text' field is required in configuration {i+1}")
                continue
                
            if not config['text'].strip():
                print(f"Error: 'text' field cannot be empty in configuration {i+1}")
                continue
                
            # Set default values for optional fields
            config.setdefault('length_scale', 1.0)
            config.setdefault('pitch_shift', 0.0)
            config.setdefault('use_pipeline', False)
            config.setdefault('output_suffix', f'_config_{i+1}')
            config.setdefault('encoding', 'utf-8')
            
            # Validate parameter ranges
            if config['length_scale'] <= 0:
                print(f"Warning: length_scale should be > 0 in config {i+1}, got {config['length_scale']}. Using default 1.0")
                config['length_scale'] = 1.0
                
            if abs(config['pitch_shift']) > 24:
                print(f"Warning: pitch_shift should be between -24 and 24 semitones in config {i+1}, got {config['pitch_shift']}. Clamping to range.")
                config['pitch_shift'] = max(-24, min(24, config['pitch_shift']))
            
            processed_configs.append(config)
        
        if processed_configs:
            print(f"Successfully loaded {len(processed_configs)} configuration(s) from '{file_path}'")
            for i, config in enumerate(processed_configs):
                print(f"  Config {i+1}: {len(config['text'])} chars, speed={config['length_scale']}, pitch={config['pitch_shift']}")
        
        return processed_configs
        
    except FileNotFoundError:
        print(f"Error: JSON file '{file_path}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{file_path}'")
        print(f"JSON Error: {e}")
        print("\nHint: If you have multiple configurations, wrap them in square brackets []")
        print("Example:")
        print("[")
        print("  { \"text\": \"First text\", ... },")
        print("  { \"text\": \"Second text\", ... }")
        print("]")
        return None
    except Exception as e:
        print(f"Error reading JSON file '{file_path}': {e}")
        return None

def adjust_audio_speed(audio_array, speed_factor, sampling_rate):
    """
    Adjust audio speed using librosa time stretching.
    speed_factor > 1.0 = faster, < 1.0 = slower
    Requires: pip install librosa
    """
    if not LIBROSA_AVAILABLE:
        print("Librosa not installed. Cannot adjust speed. Using original audio.")
        return audio_array
        
    try:
        # Convert length_scale to speed_factor (inverse relationship)
        # length_scale 0.5 = 2x faster, length_scale 2.0 = 0.5x slower
        actual_speed_factor = 1.0 / speed_factor
        
        # Use librosa to change tempo without changing pitch
        stretched_audio = librosa.effects.time_stretch(audio_array, rate=actual_speed_factor)
        return stretched_audio
    except Exception as e:
        print(f"Speed adjustment failed: {e}. Using original audio.")
        return audio_array

def adjust_audio_pitch(audio_array, pitch_shift_semitones, sampling_rate):
    """
    Adjust audio pitch using librosa pitch shifting.
    pitch_shift_semitones: positive = higher pitch, negative = lower pitch
    Range: typically -12 to +12 semitones (1 octave down to 1 octave up)
    Requires: pip install librosa
    """
    if not LIBROSA_AVAILABLE:
        print("Librosa not installed. Cannot adjust pitch. Using original audio.")
        return audio_array
        
    try:
        # Use librosa to change pitch without changing tempo
        pitched_audio = librosa.effects.pitch_shift(
            audio_array, 
            sr=sampling_rate, 
            n_steps=pitch_shift_semitones
        )
        return pitched_audio
    except Exception as e:
        print(f"Pitch adjustment failed: {e}. Using original audio.")
        return audio_array

def test_sara_tts_direct(text_input, length_scale=1.0, pitch_shift=0.0, output_filename_suffix="", output_folder="output_audio"):
    """
    Tests the BHOSAI/SARA_TTS model (VITS-based) to convert text to speech with adjustable length_scale and pitch.
    
    Args:
        text_input: Text to convert to speech
        length_scale: Speed control (1.0=normal, >1.0=slower, <1.0=faster)
        pitch_shift: Pitch adjustment in semitones (0=normal, +/-12=one octave up/down)
        output_filename_suffix: Additional suffix for output filename
        output_folder: Output directory for audio files
    """
    print(f"Loading BHOSAI/SARA_TTS model and tokenizer...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Device set to use {device}")
    else:
        print("Using CPU. Ensure PyTorch with CUDA is installed correctly for GPU usage.")

    model_name = "BHOSAI/SARA_TTS"
    try:
        # Load VITS model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name).to(device)
        print("Model and Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Please ensure you have an active internet connection to download the model, or check model name.")
        return

    print(f"\nConverting text to speech (Length Scale: {length_scale}, Pitch Shift: {pitch_shift} semitones)")
    print(f"Text preview: '{text_input[:100]}{'...' if len(text_input) > 100 else ''}'")

    try:
        os.makedirs(output_folder, exist_ok=True)
        
        filename_speed_str = str(length_scale).replace('.', '_')
        filename_pitch_str = str(pitch_shift).replace('.', '_').replace('-', 'neg')
        output_filename = f"sara_output_speed_{filename_speed_str}_pitch_{filename_pitch_str}{output_filename_suffix}.wav"
        output_path = os.path.join(output_folder, output_filename)
        print(f"Output path: {output_path}")

        # Tokenize input text
        inputs = tokenizer(text_input, return_tensors="pt").to(device)

        # Generate speech - VitsModel doesn't support .generate(), use forward pass
        with torch.no_grad():
            output = model(inputs["input_ids"])
            
        # VitsModel requires manual speed and pitch adjustments
        use_manual_adjustments = length_scale != 1.0 or pitch_shift != 0.0
        if use_manual_adjustments:
            print(f"Applying manual adjustments: speed={length_scale}, pitch={pitch_shift} semitones")

        # Extract waveform from model output
        if hasattr(output, 'waveform'):
            audio_array = output.waveform.cpu().numpy().squeeze()
        elif hasattr(output, 'audio'):
            audio_array = output.audio.cpu().numpy().squeeze()
        else:
            # Fallback for different output formats
            audio_array = output[0].cpu().numpy().squeeze()

        sampling_rate = model.config.sampling_rate

        # Apply manual adjustments (VITS models require manual speed/pitch control)
        if length_scale != 1.0:
            print(f"Applying speed adjustment: length_scale={length_scale}")
            audio_array = adjust_audio_speed(audio_array, length_scale, sampling_rate)
        
        if pitch_shift != 0.0:
            print(f"Applying pitch adjustment: {pitch_shift} semitones")
            audio_array = adjust_audio_pitch(audio_array, pitch_shift, sampling_rate)

        # Validation checks
        if audio_array.size == 0:
            print("WARNING: audio_array is empty!")
            return
        if np.isnan(audio_array).any():
            print("WARNING: audio_array contains NaN values!")
        if np.isinf(audio_array).any():
            print("WARNING: audio_array contains Inf values!")

        # Save audio file
        sf.write(output_path, audio_array, sampling_rate)

        print(f"Speech saved successfully to '{os.path.abspath(output_path)}'")
        print("You can now play this file to hear the output.")
        return output_path

    except Exception as e:
        print(f"Error during speech generation or saving: {e}")
        print("Make sure soundfile is installed and the model is compatible.")
        return None

def process_json_config(config_path):
    """
    Process a JSON configuration file and convert text to speech with specified parameters.
    Supports both single configurations and arrays of configurations.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        list: List of paths to generated audio files
    """
    # Read configuration(s) from JSON file
    configs = read_json_config(config_path)
    
    if configs is None:
        print("Failed to read JSON configuration. Exiting.")
        return []
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"PROCESSING JSON CONFIGURATION: {config_path}")
    print(f"Found {len(configs)} configuration(s) to process")
    print(f"{'='*60}")
    
    for i, config in enumerate(configs, 1):
        print(f"\n--- Processing Configuration {i}/{len(configs)} ---")
        
        # Extract parameters
        text = config['text']
        length_scale = config['length_scale']
        pitch_shift = config['pitch_shift']
        use_pipeline = config['use_pipeline']
        output_suffix = config['output_suffix']
        
        # Ensure unique output suffix for multiple configs
        if len(configs) > 1 and not output_suffix.endswith(f'_{i}'):
            output_suffix = f"{output_suffix}_{i}"
        
        print(f"Text length: {len(text)} characters")
        print(f"Length scale: {length_scale} ({'slower' if length_scale > 1 else 'faster' if length_scale < 1 else 'normal'} speed)")
        print(f"Pitch shift: {pitch_shift} semitones ({'higher' if pitch_shift > 0 else 'lower' if pitch_shift < 0 else 'normal'} pitch)")
        print(f"Method: {'Pipeline' if use_pipeline else 'Direct model'}")
        
        # Generate speech
        output_path = test_sara_tts_direct(
            text, 
            length_scale=length_scale, 
            pitch_shift=pitch_shift, 
            output_filename_suffix=output_suffix
        )
        
        results.append(output_path)
    
    return results

if __name__ == "__main__":
    # Main configuration file
    config_file = "config.json"
    
    if os.path.exists(config_file):
        results = process_json_config(config_file)
        
        # Summary
        print(f"\n{'='*80}")
        print("PROCESSING SUMMARY")
        print(f"{'='*80}")
        
        success_count = sum(1 for r in results if r is not None)
        print(f"Successfully processed: {success_count}/{len(results)} configurations")
        
        for i, result in enumerate(results, 1):
            status = "✓ SUCCESS" if result else "✗ FAILED"
            print(f"{status}: Configuration {i}")
            if result:
                print(f"    Output: {result}")
    else:
        print(f"Configuration file '{config_file}' not found.")
        print("Please create a config.json file with one of these formats:")
        print("\nSingle configuration:")
        print("""{
            "text": "Your text here",
            "length_scale": 1.0,
            "pitch_shift": 0.0,
            "use_pipeline": false,
            "output_suffix": "_custom"
        }""")
        print("\nMultiple configurations:")
        print("""[
            {
                "text": "First text",
                "length_scale": 1.0,
                "pitch_shift": 0.0,
                "use_pipeline": false,
                "output_suffix": "_first"
            },
            {
                "text": "Second text", 
                "length_scale": 1.2,
                "pitch_shift": -2.0,
                "use_pipeline": false,
                "output_suffix": "_second"
            }
        ]""")