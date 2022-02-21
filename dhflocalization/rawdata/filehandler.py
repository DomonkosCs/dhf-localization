from pathlib import Path


class FileHandler:
    def __init__(self) -> None:
        pass

    def convert_path_to_absolute(self, relative_path):
        base_path = Path(__file__).parent
        file_path = (base_path / relative_path).resolve()
        return file_path
