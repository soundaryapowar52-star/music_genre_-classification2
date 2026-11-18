"""
download_dataset.py
-------------------
Helper script to download the GTZAN Music Genre dataset, or generate
synthetic mock audio for all 10 genres to test the entire pipeline in seconds.
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

DATA_DIR = "data/genres_original"


def print_instructions():
    print("""
================================================================================
GTZAN Dataset Download Instructions
================================================================================
The GTZAN dataset contains 1,000 audio tracks (100 per genre, 30 seconds each,
sampled at 22,050 Hz) across 10 musical genres.

Option 1: Using Kaggle CLI (Recommended)
----------------------------------------
1. Install kaggle:
       pip install kaggle
2. Download GTZAN dataset:
       kaggle datasets download -d andersonbosa/gtzan-dataset-music-genre-classification
3. Extract to the 'data' folder:
       tar -xvzf gtzan-dataset-music-genre-classification.zip -C data/

Option 2: Direct Download via Python (kagglehub)
------------------------------------------------
    pip install kagglehub
    python -c "import kagglehub; print(kagglehub.dataset_download('andersonbosa/gtzan-dataset-music-genre-classification'))"

Option 3: Fast Pipeline Test (Generate Mock GTZAN Dataset in 5 Seconds)
-----------------------------------------------------------------------
If you want to verify that the code, feature extraction, CNN model training,
and prediction scripts all work immediately:
       python download_dataset.py --generate-mock

This generates clean test audio for all 10 genres without downloading 1.2 GB.
================================================================================
""")


def generate_mock_dataset(target_dir=DATA_DIR, tracks_per_genre=4, duration_sec=30, sr=22050):
    """
    Generates synthetic harmonic, percussive, and melodic audio tracks
    for all 10 genres for instant end-to-end testing of the pipeline.
    """
    print(f"Creating mock GTZAN audio dataset inside '{target_dir}'...")
    os.makedirs(target_dir, exist_ok=True)

    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    # Distinct frequency characteristics and rhythmic modulations per genre
    genre_params = {
        "blues": (196.0, 12, 0.4),      # G3 note, slow swing
        "classical": (440.0, 4, 0.2),   # A4 concert pitch, orchestral swells
        "country": (293.66, 8, 0.5),    # D4 note, strumming tempo
        "disco": (120.0, 16, 0.8),      # Kick beat tempo
        "hiphop": (65.4, 8, 0.9),       # Sub-bass 808 tempo
        "jazz": (329.63, 6, 0.3),       # E4 minor swing
        "metal": (82.4, 32, 0.8),       # Low E heavy distortion
        "pop": (523.25, 16, 0.6),       # C5 bright synth
        "reggae": (174.61, 8, 0.7),     # F3 off-beat skank
        "rock": (164.81, 16, 0.7),      # E3 power chord drive
    }

    for genre, (base_freq, mod_freq, mod_depth) in genre_params.items():
        genre_folder = os.path.join(target_dir, genre)
        os.makedirs(genre_folder, exist_ok=True)

        for i in range(tracks_per_genre):
            filename = os.path.join(genre_folder, f"{genre}.{i:05d}.wav")
            
            # Harmonic tones
            tone1 = np.sin(2 * np.pi * (base_freq + (i * 5)) * t)
            tone2 = 0.5 * np.sin(2 * np.pi * (base_freq * 1.5) * t)
            tone3 = 0.25 * np.sin(2 * np.pi * (base_freq * 2.0) * t)
            
            # Rhythmic envelope
            envelope = 1.0 - mod_depth * np.abs(np.sin(2 * np.pi * mod_freq * t))
            
            # Add subtle noise/timbre
            noise = np.random.normal(0, 0.02, size=len(t))
            
            audio = ((tone1 + tone2 + tone3) * envelope + noise).astype(np.float32)
            # Peak normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = (audio / max_val) * 0.85

            sf.write(filename, audio, sr)

        print(f"  [Created] {tracks_per_genre} tracks for genre '{genre}'")

    print("\nMock GTZAN dataset generated successfully!")
    print(f"Location: '{os.path.abspath(target_dir)}'")
    print("Now run: python dataset_prep.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTZAN Dataset Setup & Mock Generator")
    parser.add_argument("--generate-mock", action="store_true", help="Generate mock GTZAN audio tracks")
    args = parser.parse_args()

    if args.generate_mock:
        generate_mock_dataset()
    else:
        print_instructions()
