import ruamel.yaml
import numpy as np

from rawdata.filehandler import FileHandler


class ConfigExporter(FileHandler):
    def __init__(self) -> None:
        pass

    def export(self, payload, filename):
        data = self.__extract_variables(payload)

        relative_path = "../resources/results/" + filename + ".yaml"
        file_path = super().convert_path_to_absolute(relative_path)

        with open(
            file_path,
            "w",
        ) as file:
            ruamel.yaml.dump({"config": data, "results": ""}, file)

    def __extract_variables(self, payload):
        variables_dict = {}
        for key in payload.keys():
            if key.startswith("cfg_"):
                variable_data = (
                    payload[key]
                    if not isinstance(payload[key], np.ndarray)
                    else payload[key].tolist()
                )
                variables_dict[key] = variable_data
        return variables_dict


class YamlWriter(FileHandler):
    def __init__(self) -> None:
        pass

    @classmethod
    def updateFile(cls, payload, filename):

        relative_path = "../resources/results/" + filename + ".yaml"
        file_path = super().convert_path_to_absolute(cls, relative_path)

        with open(file_path, "r") as yamlfile:
            cur_yaml = ruamel.yaml.safe_load(yamlfile)  # Note the safe_load
            cur_yaml["results"] = payload

        if cur_yaml:
            with open(file_path, "w") as yamlfile:
                yaml = ruamel.yaml.YAML()
                yaml.indent(mapping=5, sequence=5, offset=3)
                yaml.dump(cur_yaml, yamlfile)  # Also note the safe_dump
