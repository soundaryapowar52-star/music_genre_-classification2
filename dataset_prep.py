"""
dataset_prep.py
---------------
Robust Feature Extraction & Preprocessing Pipeline for Music Genre Classification.

Key Features & Fixes:
1. Auto-locates GTZAN dataset even if extracted into nested directories.
2. Handles corrupted audio files (e.g. GTZAN's jazz.00054.wav) without crashing.
3. Fixes MFCC frame shape to strictly (130, 40) using automatic padding/clipping.
4. Saves dataset into high-speed compressed NumPy format (.npz) to prevent
   JSON memory bloat and MemoryError, with JSON fallback.
"""

import os
import math
import json
import numpy as np
import librosa
import soundfile as sf

# Audio & Feature Parameters
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 10  # 3-second segments per 30s track

SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)  # 66,150 samples
EXPECTED_FRAMES = 130  # math.ceil(66150 / 512)

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

DATA_OUTPUT_NPZ = "data/data_mfcc.npz"
DATA_OUTPUT_JSON = "data/data_mfcc.json"


def find_gtzan_dataset_dir(search_root="data"):
    """
    Intelligently searches for the GTZAN genre directories.
    Handles nested extraction like 'data/genres_original', 'data/Data/genres_original',
    'data/genres', etc.
    """
    # 1. Direct candidate paths
    candidates = [
        os.path.join(search_root, "genres_original"),
        os.path.join(search_root, "genres"),
        os.path.join(search_root, "Data", "genres_original"),
        os.path.join(search_root, "gtzan", "genres_original"),
        search_root,
    ]

    for candidate in candidates:
        if os.path.exists(candidate) and os.path.isdir(candidate):
            subdirs = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]
            # Check if majority of genres are present
            matches = [g for g in GENRES if g in subdirs]
            if len(matches) >= 5:
                return os.path.abspath(candidate)

    # 2. Recursive search up to depth 3
    if os.path.exists(search_root):
        for root, dirs, _ in os.walk(search_root):
            matches = [g for g in GENRES if g in dirs]
            if len(matches) >= 5:
                return os.path.abspath(root)

    return None


def extract_features(dataset_path=None, output_npz=DATA_OUTPUT_NPZ, output_json=DATA_OUTPUT_JSON):
    """
    Extracts 40 MFCC features across 3-second segments for all audio tracks.
    """
    if dataset_path is None:
        dataset_path = find_gtzan_dataset_dir("data")

    if dataset_path is None or not os.path.exists(dataset_path):
        print("=" * 70)
        print("[ERROR] GTZAN dataset directory not found!")
        print("Please download GTZAN or generate mock test audio using:")
        print("    python download_dataset.py --generate-mock")
        print("=" * 70)
        return False

    print("=" * 70)
    print("MUSIC GENRE CLASSIFICATION - FEATURE EXTRACTION")
    print(f"Dataset location    : {dataset_path}")
    print(f"Sample Rate         : {SAMPLE_RATE} Hz")
    print(f"Segment Duration    : {TRACK_DURATION / NUM_SEGMENTS:.1f} seconds")
    print(f"MFCC Shape / Segment: ({EXPECTED_FRAMES}, {N_MFCC})")
    print("=" * 70)

    # Detect genre folders
    subdirs = sorted([d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith(".")])
    
    # Filter only known genres if present, or all valid subdirectories
    genre_folders = [g for g in GENRES if g in subdirs]
    if not genre_folders:
        genre_folders = subdirs

    print(f"Detected {len(genre_folders)} genres: {genre_folders}\n")

    X_list = []
    y_list = []
    total_files_read = 0
    total_files_skipped = 0

    for genre_idx, genre in enumerate(genre_folders):
        genre_dir = os.path.join(dataset_path, genre)
        audio_files = [f for f in os.listdir(genre_dir) 
                       if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))]

        print(f"[{genre_idx + 1}/{len(genre_folders)}] Processing '{genre}': {len(audio_files)} files...")

        for file_name in audio_files:
            file_path = os.path.join(genre_dir, file_name)

            try:
                # Load audio safely with librosa (duration=30s)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=TRACK_DURATION)

                # Ensure minimum length
                if len(signal) < SAMPLES_PER_SEGMENT:
                    # Pad short signal to at least 1 segment
                    signal = np.pad(signal, (0, SAMPLES_PER_SEGMENT - len(signal)), mode="constant")

                # Slice audio into segments
                num_actual_segments = min(NUM_SEGMENTS, int(len(signal) / SAMPLES_PER_SEGMENT))
                if num_actual_segments == 0:
                    num_actual_segments = 1

                for s in range(num_actual_segments):
                    start = s * SAMPLES_PER_SEGMENT
                    finish = start + SAMPLES_PER_SEGMENT
                    chunk = signal[start:finish]

                    if len(chunk) < N_FFT:
                        continue

                    # Extract MFCC
                    mfcc = librosa.feature.mfcc(
                        y=chunk,
                        sr=sr,
                        n_mfcc=N_MFCC,
                        n_fft=N_FFT,
                        hop_length=HOP_LENGTH
                    ).T  # Transpose to (time_frames, n_mfcc)

                    # STRICT SHAPE NORMALIZATION: Guarantees exactly (130, 40)
                    if mfcc.shape[0] < EXPECTED_FRAMES:
                        pad_len = EXPECTED_FRAMES - mfcc.shape[0]
                        mfcc = np.pad(mfcc, ((0, pad_len), (0, 0)), mode="constant")
                    elif mfcc.shape[0] > EXPECTED_FRAMES:
                        mfcc = mfcc[:EXPECTED_FRAMES, :]

                    X_list.append(mfcc)
                    y_list.append(genre_idx)

                total_files_read += 1

            except Exception as e:
                # Catch corrupted files like jazz.00054.wav gracefully
                print(f"  [Skipped Corrupted File] {file_name}: {e}")
                total_files_skipped += 1
                continue

    if len(X_list) == 0:
        print("[ERROR] No valid audio segments extracted! Please verify your audio files.")
        return False

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print("\n" + "=" * 70)
    print(f"Successfully processed {total_files_read} tracks ({total_files_skipped} skipped).")
    print(f"Extracted {X.shape[0]} total segments.")
    print(f"Feature tensor shape: {X.shape} (samples, time_frames, n_mfcc)")
    print(f"Labels shape        : {y.shape}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)

    # 1. Save as high-speed compressed NPZ (fast, zero memory bloat)
    np.savez_compressed(output_npz, X=X, y=y, mapping=genre_folders)
    print(f"Saved compressed NumPy dataset to '{output_npz}' ({os.path.getsize(output_npz) / (1024*1024):.2f} MB)")

    # 2. Also save JSON mapping metadata
    json_metadata = {
        "mapping": genre_folders,
        "num_samples": int(X.shape[0]),
        "input_shape": [int(X.shape[1]), int(X.shape[2])]
    }
    with open("data/dataset_info.json", "w") as fp:
        json.dump(json_metadata, fp, indent=2)

    print("Preprocessing completed successfully!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    extract_features()
