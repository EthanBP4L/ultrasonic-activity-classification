
import os
import re


def partition_leave_one_out(file_paths, participant_id):
    """
    Partition recordings by the specified participant ID.
    """
    pattern = re.compile(
        re.escape(str(participant_id)),
        re.IGNORECASE
    )

    train_files = []
    test_files = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)

        if pattern.search(filename):
            test_files.append(file_path)
        else:
            train_files.append(file_path)

    if not test_files:
        raise ValueError(
            f"No recordings found for participant: {participant_id}"
        )

    if not train_files:
        raise ValueError("No training recordings remain.")

    return train_files, test_files
