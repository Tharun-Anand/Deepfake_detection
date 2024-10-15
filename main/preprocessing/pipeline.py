class PipeLine:
    def __init__(self,components):
        self.components = components
    def process(self,data):
        for c in self.components:
            data = c.process(data)
        return data