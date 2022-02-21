import ruamel.yaml
import numpy as np


class ConfigExporter:
    def __init__(self) -> None:
        pass

    def export(self, payload, filename):
        data = self.__extract_variables(payload)

        path = r"/Users/domonkoscsuzdi/dhf_loc/dhflocalization/resources/results/{}.yaml".format(
            filename
        )
        with open(
            path,
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


class YamlWriter:
    def __init__(self) -> None:
        pass

    @classmethod
    def updateFile(cls, payload, filename):
        path = r"/Users/domonkoscsuzdi/dhf_loc/dhflocalization/resources/results/{}.yaml".format(
            filename
        )
        with open(path, "r") as yamlfile:
            cur_yaml = ruamel.yaml.safe_load(yamlfile)  # Note the safe_load
            cur_yaml["results"] = payload

        if cur_yaml:
            with open(path, "w") as yamlfile:
                yaml = ruamel.yaml.YAML()
                yaml.indent(mapping=5, sequence=5, offset=3)
                yaml.dump(cur_yaml, yamlfile)  # Also note the safe_dump
