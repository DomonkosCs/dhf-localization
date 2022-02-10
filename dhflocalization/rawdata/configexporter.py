import yaml


class ConfigExporter:
    def __init__(self) -> None:
        self.data = {}

    def add(self, key: str, value):
        self.data[key] = value

    def export(self, filename):
        path = r"/Users/domonkoscsuzdi/dhf_loc/dhflocalization/resources/results/{}.yaml".format(
            filename
        )
        with open(
            path,
            "w",
        ) as file:
            yaml.dump(self.data, file)
