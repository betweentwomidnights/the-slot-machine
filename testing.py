import time
from musiclang_predict import MusicLangPredictor

# Parameters for music generation
nb_tokens = 1024  # Number of tokens to generate
temperature = 0.9  # Generation temperature
top_p = 1.0  # Top-p for nucleus sampling
seed = 10000  # Seed for reproducibility

# Initialize MusicLangPredictor
ml = MusicLangPredictor('musiclang/musiclang-v2')

# Generate music
start = time.time()
score = ml.predict(
    nb_tokens=nb_tokens,
    temperature=temperature,
    topp=top_p,
    rng_seed=seed  # Change here to change result, or set to 0 to unset seed
)
end = time.time()
print(f"Generated in {end - start} seconds")

# Save to MIDI
score.to_midi('test.mid')
print("MIDI file saved as 'test.mid'")
