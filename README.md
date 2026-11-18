# 🎵 Music Genre Classification using Deep Learning (CNN & Librosa)

A robust, production-ready Audio Machine Learning pipeline that predicts the musical genre of an audio file using **Librosa**, **Mel-Frequency Cepstral Coefficients (MFCC)**, and a **2D Convolutional Neural Network (CNN)** built with **TensorFlow / Keras**.

---

## 🛠️ Errors Diagnosed & Fixed

| Issue / Error | Root Cause | Solution Applied |
| :--- | :--- | :--- |
| **`ValueError: Only instances of Layer can be added to Sequential`** | In Keras 3 (TensorFlow 2.16+), `layers.Input()` returns a `KerasTensor` rather than a Layer, breaking `Sequential([layers.Input(...)])`. | Migrated to the **Keras Functional API** (`Model(inputs, outputs)`), which is 100% compatible across Keras 2 and Keras 3. |
| **`MemoryError` / Process Freeze during JSON export** | Storing 10,000 segments with 52 million float elements in an indented `.json` file consumes ~1.8 GB disk space and ~8 GB RAM. | Replaced with compressed NumPy format (`data_mfcc.npz`), which takes only **~40 MB**, writes in 2 seconds, and eliminates memory crashes. |
| **`ValueError: operands could not be broadcast (129, 40) vs (130, 40)`** | Librosa STFT padding variations occasionally output 129 or 131 time frames instead of 130, causing shape mismatches or dropped audio. | Added strict frame normalization: automatically pads to 130 if `<130` or slices if `>130`. |
| **`LibsndfileError` on `jazz.00054.wav`** | GTZAN includes a well-known damaged track (`jazz.00054.wav`) that crashes audio decoders. | Added robust `try...except` file-level exception handling to log and skip damaged files. |
| **`FileNotFoundError` on nested GTZAN downloads** | Extracting GTZAN from Kaggle often nests files under `data/Data/genres_original` or `data/genres`. | Added recursive auto-discovery (`find_gtzan_dataset_dir()`) to detect the genre folders automatically. |
| **Notebook crash during "Run All"** | Missing dataset or missing `data_mfcc.json` crashes subsequent cells if executed sequentially. | Notebook now automatically generates mock harmonic data if GTZAN is missing, ensuring **all cells run end-to-end without errors**. |
| **Windows `PermissionError: [WinError 32]` in Streamlit** | Windows file locks occur when `NamedTemporaryFile` remains open while `librosa.load` accesses it. | Explicitly closed file handles before loading, then cleanly removed temp files in a `finally` block. |

---

## 📂 Project Architecture

```
music_genre_classification/
├── data/
│   ├── genres_original/       # Audio files categorized by genre
│   └── data_mfcc.npz          # Compressed preprocessed MFCC features
├── models/
│   ├── music_genre_cnn.keras  # Saved best trained CNN model
│   └── genre_mapping.json     # Saved list of genre labels
├── plots/
│   ├── training_curves.png    # Training vs Validation loss & accuracy
│   └── confusion_matrix.png   # Multi-class confusion matrix
├── dataset_prep.py            # Audio slicing & MFCC extraction pipeline
├── download_dataset.py        # Dataset downloader & instant mock audio generator
├── train.py                   # CNN training script (80/20 train/val split)
├── predict.py                 # CLI prediction script with confidence scores & plot
├── app.py                     # Streamlit web demo with audio player & spectrograms
├── music_genre_classification.ipynb # Complete, self-contained Jupyter Notebook
├── requirements.txt           # Verified non-conflicting dependencies
└── README.md                  # Documentation and guide
```

---

## 🚀 How to Run (Step-by-Step)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Audio Data
Choose either:
* **Instant 5-second test (generates clean synthetic audio across all 10 genres):**
  ```bash
  python download_dataset.py --generate-mock
  ```
* **Or download the full 1.2 GB GTZAN dataset from Kaggle:**
  ```bash
  kaggle datasets download -d andersonbosa/gtzan-dataset-music-genre-classification
  tar -xvzf gtzan-dataset-music-genre-classification.zip -C data/
  ```

### Step 3: Extract Features
Runs the audio slicing (3s windows) and 40-coefficient MFCC extraction:
```bash
python dataset_prep.py
```
*Output: Saves compressed features to `data/data_mfcc.npz`.*

### Step 4: Train the 2D CNN Model
Trains the model on an **80/20 train/validation split**:
```bash
python train.py
```
*Outputs: Saves `models/music_genre_cnn.keras`, plots training curves in `plots/training_curves.png`, and saves `plots/confusion_matrix.png`.*

### Step 5: Classify Any Song (CLI)
Classify any song file (`.wav`, `.mp3`, `.ogg`) and view confidence scores:
```bash
python predict.py "data/genres_original/rock/rock.00000.wav" --plot
```

### Step 6: Launch the Interactive Web App (Streamlit)
```bash
streamlit run app.py
```
Upload any audio file to listen, visualize its **Mel-Spectrogram**, and inspect real-time genre confidence gauges!

### Step 7: Jupyter Notebook
Run the notebook end-to-end:
```bash
jupyter notebook music_genre_classification.ipynb
```
