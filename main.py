
import os
import argparse
import numpy as np

from data_processing import process_activity_file
from in_domain import partition_in_domain
from leave_one_out import partition_leave_one_out

from model import create_model, train_model

from sklearn.utils import shuffle

def get_audio_files(data_path):
    """
    Recursively collect all WAV file paths.
    """
    file_paths = []

    for root, dirs, files in os.walk(data_path):
        for filename in sorted(files):

            if not filename.lower().endswith(".wav"):
                continue

            file_path = os.path.join(root, filename)
            file_paths.append(file_path)

    return sorted(file_paths)


def load_dataset(
    file_paths,
    augment=False,
    augmentation_methods=None
):
    """
    Process a list of audio files into features and labels.
    """
    total_features = []
    total_labels = []
    total_individual_labels = []

    for file_path in file_paths:

        print(f"\nProcessing: {file_path}")

        individual_labels, labels, features = (
            process_activity_file(
                file_path,
                augment=augment,
                augmentation_methods=augmentation_methods
            )
        )

        total_features.extend(features)
        total_labels.extend(labels)
        total_individual_labels.extend(individual_labels)

    X = np.asarray(total_features)
    y = np.asarray(total_labels)
    individuals = np.asarray(total_individual_labels)

    if not (len(X) == len(y) == len(individuals)):
        raise ValueError(
            "Features, labels, and individual labels do not match!"
        )

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

    parser.add_argument(
        "--experiment",
        choices=["in_domain", "leave_one_out"],
        default="in_domain",
        help="Dataset partitioning method"
    )

    parser.add_argument(
        "--participant_id",
        type=str,
        default=None,
        help="Participant held out for leave-one-out testing"
    )

    parser.add_argument(
        "--augmentation_methods",
        nargs="+",
        choices=[
            "time_stretch",
            "noise",
            "pitch_shift",
            "carrier_205",
            "carrier_195"
        ],
        default=None,
        help="Augmentations applied to training data only"
    )

    args = parser.parse_args()

    # Step 1: Find all recordings
    file_paths = get_audio_files(args.data_path)

    if len(file_paths) < 2:
        raise ValueError(
            "At least two WAV files are required."
        )

    print(f"\nFound {len(file_paths)} WAV files.")

    # Step 2: Partition recordings
    if args.experiment == "in_domain":

        train_files, test_files = partition_in_domain(
            file_paths
        )

    elif args.experiment == "leave_one_out":

        if args.participant_id is None:
            parser.error(
                "--participant_id is required for leave_one_out"
            )

        train_files, test_files = partition_leave_one_out(
            file_paths,
            args.participant_id
        )

    print(f"\nExperiment: {args.experiment}")
    print(f"Training recordings: {len(train_files)}")
    print(f"Testing recordings: {len(test_files)}")

    # Step 3: Extract training features
    print("\nProcessing Training Dataset")
    print("---------------------------")

    X_train, y_train, individuals_train = load_dataset(
        train_files,
        augment=bool(args.augmentation_methods),
        augmentation_methods=args.augmentation_methods
    )

    # Step 4: Extract testing features
    print("\nProcessing Testing Dataset")
    print("--------------------------")

    X_test, y_test, individuals_test = load_dataset(
        test_files,
        augment=False,
        augmentation_methods=None
    )

    # Shuffle training data while keeping labels aligned
    X_train, y_train, individuals_train = shuffle(
        X_train,
        y_train,
        individuals_train,
        random_state=42
    )

    # Trim each activity to 16 time steps
    X_train = X_train[:, :16, :]
    X_test = X_test[:, :16, :]

    # Step 5: Display dataset information
    print("\nDataset Processing Complete")
    print("---------------------------")

    print(f"Training features: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Training individuals: {individuals_train.shape}")

    print(f"\nTesting features: {X_test.shape}")
    print(f"Testing labels: {y_test.shape}")
    print(f"Testing individuals: {individuals_test.shape}")

        # Create a fresh model for this experiment
    model = create_model(
        input_shape=X_train.shape[1:],
        num_classes=11
    )

    model.summary()

    # Train the model
    history = train_model(
        model,
        X_train,
        y_train
    )

    # Evaluate on held-out testing data
    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print("\nModel Evaluation Complete")
    print("-------------------------")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")



if __name__ == "__main__":
    main()


