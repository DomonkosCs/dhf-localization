import pickle
from datetime import datetime
from pathlib import Path


class resultExporter:
    def __init__(self) -> None:
        pass

    @classmethod
    def save(cls, *datas, prefix="") -> str:
        data_list = []
        for data in datas:
            data_list.append(data)
        time = datetime.now()

        file_name = time.strftime("%y-%m-%dT%H%M%S%z")
        base_path = Path(__file__).parent
        relative_path = "../resources/results/" + file_name + ".p"
        file_path = (base_path / relative_path).resolve()

        if prefix != "":
            prefix += "-"
        pickle.dump(data_list, open(file_path, "wb"))
        return file_name


class resultLoader:
    def __init__(self) -> None:
        pass

    @classmethod
    def load(cls, file_name):
        base_path = Path(__file__).parent
        relative_path = "../resources/results/" + file_name + ".p"
        file_path = (base_path / relative_path).resolve()

        loaded_data = pickle.load(open(file_path, "rb"))
        return loaded_data
