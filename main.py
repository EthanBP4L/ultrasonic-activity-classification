
import os
import argparse
import numpy as np

from data_processing import process_activity_file


def load_dataset(data_path):
    total_features = []
    total_labels = []
    total_individual_labels = []

    # Walk through every folder in the dataset
    for root, dirs, files in os.walk(data_path):

        for filename in sorted(files):

            # Only process WAV audio files
            if not filename.lower().endswith(".wav"):
                continue

            # Construct the complete file path
            file_path = os.path.join(root, filename)

            print(f"\nProcessing: {file_path}")

            # Call the processing function
            individual_labels, labels, features = (
                process_activity_file(file_path)
            )

            # Add this recording's results to the dataset
            total_features.extend(features)
            total_labels.extend(labels)
            total_individual_labels.extend(individual_labels)

    # Convert the collected results into NumPy arrays
    X = np.asarray(total_features)
    y = np.asarray(total_labels)
    individuals = np.asarray(total_individual_labels)

    return X, y, individuals


def main():
    parser = argparse.ArgumentParser(
        description="Ultrasonic Activity Recognition"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )

    args = parser.parse_args()

    # Process the dataset
    X, y, individuals = load_dataset(args.data_path)

    # Display dataset information
    print("\nDataset Processing Complete")
    print("---------------------------")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Individual labels shape: {individuals.shape}")

    if len(X) != len(y) or len(y) != len(individuals):
        raise ValueError(
            "Features, labels, and individual labels do not match!"
        )


if __name__ == "__main__":
    main()
