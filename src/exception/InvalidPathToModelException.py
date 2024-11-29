class InvalidPathToModelException(Exception):
    def __init__(self):
        super().__init__("file extension should be .pkl")
