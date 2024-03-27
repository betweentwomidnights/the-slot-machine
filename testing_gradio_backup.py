import gradio as gr
from musiclang_predict import MusicLangPredictor
import random
import subprocess
import os

def generate_music(seed, use_chords, chord_progression):
    if seed == "":
        seed = random.randint(1, 10000)

    ml = MusicLangPredictor('musiclang/musiclang-v2')

    try:
        seed = int(seed)
    except ValueError:
        seed = random.randint(1, 10000)

    nb_tokens = 1024
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
    mp3_filename = midi_filename.replace(".mid", ".mp3")

    score.to_midi(midi_filename, tempo=110, time_signature=(4, 4))

    subprocess.run(["fluidsynth", "-ni", "font.sf2", midi_filename, "-F", midi_filename.replace(".mid", ".wav"), "-r", "44100"])
    subprocess.run(["ffmpeg", "-i", midi_filename.replace(".mid", ".wav"), "-acodec", "libmp3lame", "-y", mp3_filename])

    os.remove(midi_filename.replace(".mid", ".wav"))

    return mp3_filename

iface = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Seed (leave blank for random)", value=""),
        gr.Checkbox(label="Control Chord Progression", value=False),
        gr.Textbox(label="Chord Progression (e.g., Am CM Dm E7 Am)", visible=True)
    ],
    outputs=gr.Audio(label="Generated Music"),
    title="Music Generation Slot Machine",
    description="Enter a seed to generate music or leave blank for a random tune! Optionally, control the chord progression."
)

iface.launch()
