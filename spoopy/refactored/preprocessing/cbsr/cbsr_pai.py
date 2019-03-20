from refactored.preprocessing.pai.pai import BasePaiConfig, Pai


class CbsrPaiConfig(BasePaiConfig):
    """
    This is an example of the BasePaiConfig class for the CBSR dataset
    """

    def __init__(self):
        self.pai_dict = {}

    def get_pai_alias(self, key) -> str:
        items = self.pai_dict.items()
        for item in items:
            if key in item[1]:
                return item[0]

    def add_pai(self, pai: Pai) -> None:
        self.pai_dict[pai.alias] = pai.name_files
