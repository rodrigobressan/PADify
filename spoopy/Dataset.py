class Dataset:
    def __init__(self, name):
        self.types = []
        self.name = name

    def addType(self, type):
        self.types.append(type)
