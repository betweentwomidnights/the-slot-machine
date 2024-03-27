import gradio as gr
from musiclang_predict import MusicLangPredictor
import random
import subprocess
import os
import torchaudio
import torch
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pydub import AudioSegment

# Utility Functions
def peak_normalize(y, target_peak=0.97):
    return target_peak * (y / np.max(np.abs(y)))

def rms_normalize(y, target_rms=0.05):
    return y * (target_rms / np.sqrt(np.mean(y**2)))

def preprocess_audio(waveform):
    waveform_np = waveform.cpu().squeeze().numpy()  # Move to CPU before converting to NumPy
    processed_waveform_np = rms_normalize(peak_normalize(waveform_np))
    return torch.from_numpy(processed_waveform_np).unsqueeze(0).to(device)

def create_slices(song, sr, slice_duration, bpm, num_slices=5):
    song_length = song.shape[-1] / sr
    slices = []
    
    # Ensure the first slice is from the beginning of the song
    first_slice_waveform = song[..., :int(slice_duration * sr)]
    slices.append(first_slice_waveform)
    
    for i in range(1, num_slices):
        random_start = random.choice(range(int(slice_duration * sr), int(song_length * sr), int(4 * 60 / bpm * sr)))
        slice_end = random_start + int(slice_duration * sr)
        
        if slice_end > song_length * sr:
            # Wrap around to the beginning of the song
            remaining_samples = int(slice_end - song_length * sr)
            slice_waveform = torch.cat([song[..., random_start:], song[..., :remaining_samples]], dim=-1)
        else:
            slice_waveform = song[..., random_start:slice_end]
        
        if len(slice_waveform.squeeze()) < int(slice_duration * sr):
            additional_samples_needed = int(slice_duration * sr) - len(slice_waveform.squeeze())
            slice_waveform = torch.cat([slice_waveform, song[..., :additional_samples_needed]], dim=-1)
        
        slices.append(slice_waveform)
        
    return slices

def calculate_duration(bpm, min_duration=29, max_duration=30):
    single_bar_duration = 4 * 60 / bpm
    bars = max(min_duration // single_bar_duration, 1)
    
    while single_bar_duration * bars < min_duration:
        bars += 1
    
    duration = single_bar_duration * bars
    
    while duration > max_duration and bars > 1:
        bars -= 1
        duration = single_bar_duration * bars
    
    return duration

def generate_music(seed, use_chords, chord_progression, prompt_duration, musicgen_model, num_iterations, bpm):
    if seed == "":
        seed = random.randint(1, 10000)

    ml = MusicLangPredictor('musiclang/musiclang-v2')

    try:
        seed = int(seed)
    except ValueError:
        seed = random.randint(1, 10000)

    nb_tokens = 4096
    temperature = 0.9
    top_p = 1.0

    if use_chords and chord_progression.strip():
        score = ml.predict_chords(
            chord_progression,
            time_signature=(4, 4),
            temperature=temperature,
            topp=top_p,
            rng_seed=seed
        )
    else:
        score = ml.predict(
            nb_tokens=nb_tokens,
            temperature=temperature,
            topp=top_p,
            rng_seed=seed
        )

    midi_filename = f"output_{seed}.mid"
    wav_filename = midi_filename.replace(".mid", ".wav")

    score.to_midi(midi_filename, tempo=bpm, time_signature=(4, 4))

    subprocess.run(["fluidsynth", "-ni", "font.sf2", midi_filename, "-F", wav_filename, "-r", "44100"])

    # Load the generated audio
    song, sr = torchaudio.load(wav_filename)
    song = song.to(device)

    # Use the user-provided BPM value for duration calculation
    duration = calculate_duration(bpm)

    # Create slices from the song using the user-provided BPM value
    slices = create_slices(song, sr, 35, bpm, num_slices=5)

    # Load the model
    model_continue = MusicGen.get_pretrained(musicgen_model)

    # Setting generation parameters
    model_continue.set_generation_params(
        use_sampling=True,
        top_k=250,
        top_p=0.0,
        temperature=1.0,
        duration=duration,
        cfg_coef=3
    )

    all_audio_files = []

    for i in range(num_iterations):
        slice_idx = i % len(slices)
        
        print(f"Running iteration {i + 1} using slice {slice_idx}...")
        
        prompt_waveform = slices[slice_idx][..., :int(prompt_duration * sr)]
        prompt_waveform = preprocess_audio(prompt_waveform)
        
        output = model_continue.generate_continuation(prompt_waveform, prompt_sample_rate=sr, progress=True)
        output = output.cpu()  # Move the output tensor back to CPU
        
        # Make sure the output tensor has at most 2 dimensions
        if len(output.size()) > 2:
            output = output.squeeze()
        
        filename_without_extension = f'continue_{i}'
        filename_with_extension = f'{filename_without_extension}.wav'
        
        audio_write(filename_with_extension, output, model_continue.sample_rate, strategy="loudness", loudness_compressor=True)
        all_audio_files.append(f'{filename_without_extension}.wav.wav')  # Assuming the library appends an extra .wav

    # Combine all audio files
    combined_audio = AudioSegment.empty()
    for filename in all_audio_files:
        combined_audio += AudioSegment.from_wav(filename)

    combined_audio_filename = f"combined_audio_{seed}.mp3"
    combined_audio.export(combined_audio_filename, format="mp3")

    # Clean up temporary files
    os.remove(midi_filename)
    os.remove(wav_filename)
    for filename in all_audio_files:
        os.remove(filename)

    return combined_audio_filename

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

iface = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Seed (leave blank for random)", value=""),
        gr.Checkbox(label="Control Chord Progression", value=False),
        gr.Textbox(label="Chord Progression (e.g., Am CM Dm E7 Am)", visible=True),
        gr.Dropdown(label="Prompt Duration (seconds)", choices=list(range(1, 11)), value=7),
        gr.Textbox(label="MusicGen Model", value="thepatch/vanya_ai_dnb_0.1"),
        gr.Slider(label="Number of Iterations", minimum=1, maximum=10, step=1, value=3),
        gr.Slider(label="BPM", minimum=60, maximum=200, step=1, value=140)
    ],
    outputs=gr.Audio(label="Generated Music"),
    title="Music Generation Slot Machine",
    description="Enter a seed to generate music or leave blank for a random tune! Optionally, control the chord progression, prompt duration, MusicGen model, number of iterations, and BPM."
)

iface.launch()