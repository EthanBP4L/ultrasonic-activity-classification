# Ultrasonic Activity Classification

**A signal-processing and deep-learning pipeline for human activity classification using ultrasonic audio signals.**

This project investigates human activity recognition through ultrasonic sensing. Audio recordings containing ultrasonic signals are processed to identify individual activity repetitions, extract frequency-domain features, and classify activities using a regularized Bidirectional Long Short-Term Memory (BiLSTM) neural network.

The pipeline is designed for reproducible experimentation with different data augmentation techniques and evaluation strategies, including in-domain testing and leave-one-participant-out testing.

## Overview
Ultrasonic sensing offers a method for recognizing human activities by analyzing changes in high-frequency acoustic signals caused by movement.

This project implements an end-to-end workflow that:

1. Loads and preprocesses ultrasonic audio recordings.
2. Detects and segments individual activity repetitions.
3. Extracts frequency-domain representations using the Fast Fourier Transform (FFT).
4. Optionally generates augmented training samples from individual activity segments.
5. Partitions recordings into training and testing datasets.
6. Trains a regularized BiLSTM classifier.
7. Evaluates classification performance on held-out data.

The implementation is modular, allowing signal-processing techniques, augmentation strategies, and model configurations to be modified independently.

## Project Structure

```text
ultrasonic-activity-classification
├── main.py
├── data_processing.py
├── data_augmentation.py
├── in_domain.py
├── leave_one_out.py
├── model.py
├── requirements.txt
└── README.md
```

| File | Description |
|---|---|
| `main.py` | Command-line interface and experiment orchestration |
| `data_processing.py` | Audio preprocessing, activity segmentation, and FFT feature extraction |
| `data_augmentation.py` | Audio-domain augmentation methods |
| `in_domain.py` | Random 80/20 training/testing partition |
| `leave_one_out.py` | Participant-based training/testing partition |
| `model.py` | Regularized BiLSTM architecture and training |
| `requirements.txt` | Python dependencies |

## Signal Processing Pipeline

### 1. Audio Preprocessing

The pipeline processes ultrasonic audio recordings at a sampling rate of **48,000 Hz**.

Preprocessing includes:

- Resampling audio to 48 kHz when necessary.
- Applying a bandpass filter to isolate ultrasonic frequencies.
- Applying a bandstop filter to suppress the transmitted carrier frequency.
- Analyzing signal magnitude to identify activity boundaries.

For recordings centered at 20 kHz, the primary filtering configuration is:

| Parameter         | Value |
|--|--|
| Sampling rate     | 48,000 Hz |
| Bandpass range    | 19,300–20,700 Hz |
| Bandstop range    | 19,985–20,015 Hz |
| Carrier frequency | 20,000 Hz |

### 2. Activity Segmentation

Rather than treating an entire audio recording as a single training sample, the pipeline identifies individual activity repetitions.

Activity boundaries are determined using ultrasonic signal magnitude and activity-specific segmentation logic.

Each detected repetition is processed independently, allowing recordings containing multiple repetitions to contribute multiple samples to the dataset.

### 3. FFT Feature Extraction

Each valid activity segment is divided into consecutive audio frames.

The FFT is applied to each frame to extract frequency-domain magnitude features within the ultrasonic frequency band.

| Parameter          | Value |
|--|--|
| Frame duration     | 0.05 seconds |
| Samples per frame  | 2,400 |
| FFT size | 4,096   |
| Extracted frames   | 20 |
| Model input frames | 16 |
| Frequency features | 118 |

The extracted features are trimmed to 16 time steps before model training, producing an input representation of:


(16, 118)


This represents approximately 0.8 seconds of ultrasonic activity information per sample.

## Data Augmentation

The project supports audio-domain augmentation to investigate model robustness and generalization.

Augmentations are applied to **individual activity segments**, rather than entire recordings.

Each selected augmentation produces a separate sample derived from the original activity. Multiple augmentation methods do not need to be applied sequentially to the same signal.

### Supported Methods

| Augmentation | Description | Configuration |
|--|--|--|
| Temporal scaling | Changes activity duration while preserving pitch | Rate: 0.90–1.10 |
| Band-limited noise | Adds ultrasonic noise at a controlled signal-to-noise ratio | SNR: 40–50 dB |
| Pitch shifting | Shifts signal frequencies using an analytic-signal representation | ±24 Hz |
| Carrier shift (20.5 kHz) | Translates the ultrasonic carrier from 20 kHz to 20.5 kHz | +500 Hz |
| Carrier shift (19.5 kHz) | Translates the ultrasonic carrier from 20 kHz to 19.5 kHz | −500 Hz |

The augmentation implementations are located in `data_augmentation.py`.

**To prevent data leakage, augmentation is applied exclusively to training recordings.** Testing recordings are processed without augmentation.

## Deep Learning Model

The classification model uses a regularized, stacked Bidirectional LSTM architecture implemented with TensorFlow/Keras.


### Model Architecture

```text
Input: (16, 118)
       |
       v
Bidirectional LSTM
64 units per direction
Dropout: 0.20
L2 regularization
Return sequences: True
       |
       v
Bidirectional LSTM
64 units per direction
Dropout: 0.20
L2 regularization
Return sequences: False
       |
       v
Dropout (0.30)
       |
       v
Dense (11, Softmax)
       |
       v
Predicted Activity Class
```

The bidirectional architecture allows the model to learn temporal relationships in both directions across consecutive FFT frames.


### Training Configuration

| Hyperparameter          | Value |
|--|--|
| Model                   | Stacked BiLSTM |
| LSTM units              | 64 per direction, per layer |
| LSTM layers             | 2 |
| LSTM dropout            | 0.20 |
| Final dropout           | 0.30 |
| L2 regularization       | 0.0001 |
| Optimizer               | Adam |
| Learning rate           | 0.001 |
| Loss                    | Sparse categorical crossentropy |
| Batch size              | 32 |
| Maximum epochs          | 50 |
| Early stopping patience | 10 |
| Early stopping monitor  | Training loss |
| Minimum improvement     | 0.001 |
| Output classes          | 11 |

Early stopping restores the best model weights observed during training according to the monitored loss.

## Experimental Configurations

The project supports two evaluation strategies.

### In-Domain Testing

The default experiment randomly partitions audio recordings into:

- **80% training recordings**
- **20% testing recordings**

The partition uses a fixed random seed for reproducibility.

All activity repetitions extracted from a given recording remain in the same partition.

### Leave-One-Participant-Out Testing

The leave-one-out configuration evaluates generalization to a participant excluded from training.

A participant identifier is supplied through the command line. Recordings with filenames matching that identifier are assigned to the testing dataset, while the remaining recordings are used for training.

This enables experiments investigating how well a model trained on one set of participants generalizes to an unseen participant.

The current implementation evaluates one specified participant per execution.

## Installation

### Prerequisites

- Python 3.11
- A compatible Python environment
- Ultrasonic WAV recordings


### Install Dependencies

All required Python packages are listed in `requirements.txt`.

## Usage

The project is executed through `main.py`.

### Default In-Domain Experiment

```bash
python main.py --data_path ./data
```

Performs the default 80/20 recording-level partition without augmentation.

### In-Domain Experiment with Augmentation

```bash
python main.py --data_path ./data --experiment in_domain --augmentation_methods time_stretch noise pitch_shift
```

Each selected augmentation generates an independent training sample for every eligible activity repetition.

### Leave-One-Out Experiment

```bash
python main.py --data_path ./data --experiment leave_one_out --participant_id Ethan
```

Excludes recordings matching the participant identifier `Ethan` from training and uses them for testing.

### Leave-One-Out with Augmentation

```bash
python main.py --data_path ./data --experiment leave_one_out --participant_id Ethan --augmentation_methods time_stretch noise
```

### Command-Line Arguments

| Argument | Description | Default |
|--|--|--|
| `--data_path` | Path to the dataset directory | Required |
| `--experiment` | `in_domain` or `leave_one_out` | `in_domain` |
| `--participant_id` | Identifier used to select the held-out participant | `None` |
| `--augmentation_methods` | One or more supported augmentation methods | None |

Available augmentation method identifiers:

```text
time_stretch
noise
pitch_shift
carrier_205
carrier_195
```

## Reproducibility

The pipeline is structured to support consistent experimental comparisons.

Key considerations include:

- Partitioning recordings before feature extraction and augmentation.
- Preventing augmented versions of training samples from entering the test set.
- Using a fixed random seed for the in-domain partition.
- Maintaining consistent FFT extraction and model configurations across experiments.
- Keeping the model architecture and training hyperparameters consistent when comparing augmentation methods.

These controls help isolate the effects of individual augmentation techniques and evaluation strategies.

## Research Objectives

This project focuses on understanding the effectiveness and generalization capabilities of ultrasonic activity recognition models.

Primary research directions include:

- Classification of human activities from ultrasonic frequency-domain signals.
- Temporal feature learning using LSTM.
- Robustness to differences between participants.
- Generalization across recording conditions and environments.
- Evaluation of audio-domain data augmentation strategies.
- Comparison of in-domain and participant-held-out performance.

---

**Ultrasonic Activity Classification** — Exploring human activity recognition through ultrasonic signal processing and deep learning.