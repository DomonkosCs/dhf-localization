from os import path
import pickle
from datetime import date, datetime


class resultExporter():
    def __init__(self) -> None:
        pass

    @classmethod
    def save(cls, *datas, prefix=''):
        data_list = []
        for data in datas:
            data_list.append(data)
        time = datetime.now()
        path = 'resources/results/'
        file_name = time.strftime('%m-%d-%H_%M')
        extension = '.p'
        if prefix is not '':
            prefix += '-'
        pickle.dump(data_list, open(path+prefix+file_name+extension, "wb"))


class resultLoader():
    def __init__(self) -> None:
        pass

    @classmethod
    def load(cls, file_name, path=None):
        extension = '.p'
        if path is None:
            path = 'resources/results/'
        loaded_data = pickle.load(open(path+file_name+extension, "rb"))
        return loaded_data
