import yaml
import numpy as np


class ConfigExporter:
    def __init__(self) -> None:
        self.data = {}

    def export(self, payload, filename):
        self.data = self.__extract_variables(payload)

        path = r"/Users/domonkoscsuzdi/dhf_loc/dhflocalization/resources/results/{}.yaml".format(
            filename
        )
        with open(
            path,
            "w",
        ) as file:
            yaml.dump(self.data, file)

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
