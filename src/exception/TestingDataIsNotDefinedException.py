class TestingDataIsNotDefinedException(Exception):
    def __init__(self):
        super().__init__("TestingDataIsNotDefinedException")
