import pickle
from datetime import datetime

from ..rawdata.filehandler import FileHandler


class ResultExporter(FileHandler):
    @classmethod
    def save(cls, *datas, prefix="") -> str:
        data_list = []
        for data in datas:
            data_list.append(data)
        time = datetime.now()

        file_name = time.strftime("%y-%m-%dT%H%M%S%z")
        relative_path = "../resources/results/" + file_name + ".p"
        file_path = super().convert_path_to_absolute(cls, relative_path)

        if prefix != "":
            prefix += "-"
        pickle.dump(data_list, open(file_path, "wb"))
        return file_name


class ResultLoader(FileHandler):
    @classmethod
    def load(cls, file_name):
        relative_path = "../resources/results/" + file_name + ".p"
        file_path = super().convert_path_to_absolute(cls, relative_path)

        loaded_data = pickle.load(open(file_path, "rb"))
        return loaded_data[0]
