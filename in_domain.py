
from sklearn.model_selection import train_test_split


def partition_in_domain(file_paths, test_size=0.2, seed=42):
    """
    Split recording paths into training and testing sets.

    All activity repetitions from a recording remain
    in the same partition.
    """
    train_files, test_files = train_test_split(
        file_paths,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    return train_files, test_files
