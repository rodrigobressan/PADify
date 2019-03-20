class Type:
    def __init__(self, name):
        self.items = []
        self.name = name

    def addItem(self, id):
        self.items.append(id)
