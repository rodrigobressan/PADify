from refactored.preprocessing.pai.pai import BasePaiConfig, Pai


class DefaultPaiConfig(BasePaiConfig):
    """
    This is an example of the BasePaiConfig class for the Replay Attack dataset
    """

    def __init__(self):
        self.pai_dict = {}

    def get_pai_alias(self, key) -> str:
        clean_name = key.split('_')[1]
        items = self.pai_dict.items()
        for item in items:
            if clean_name in item[1]:
                return item[0]

    def add_pai(self, pai: Pai) -> None:
        self.pai_dict[pai.alias] = pai.name_files
