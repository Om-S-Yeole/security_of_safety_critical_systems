from pathlib import Path

import yaml


def load_yaml(file_path: str | Path):
    if not isinstance(file_path, (str, Path)):
        raise TypeError(
            f"Expected type of file_path is (str, Path). Got {type(file_path)}"
        )

    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        return data