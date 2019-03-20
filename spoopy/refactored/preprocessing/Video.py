class Video():
    def __init__(self,
                 path: str,
                 name: str,
                 person: str,
                 subset: str,
                 is_attack: bool,
                 pai: str = None):
        self.path = path
        self.name = name
        self.person = person
        self.subset = subset
        self.is_attack = is_attack
        self.pai = pai

    def get_label(self) -> str:

        if self.is_attack:
            return "attack"
        else:
            return "real"

    def get_frame_prefix(self) -> str:
        """
        Return the name for frame's prefix

        Format: <PERSON>_<NAME>
        :return: str containing the name for the prefix
        """
        return self.clean_name()

    def clean_name(self):
        """
        Remove extension from name
        :return:
        """
        return self.name.split('.')[0]
