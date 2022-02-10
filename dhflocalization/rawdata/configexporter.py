import yaml


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
        return {key: payload[key] for key in payload.keys() if key.startswith("cfg_")}
