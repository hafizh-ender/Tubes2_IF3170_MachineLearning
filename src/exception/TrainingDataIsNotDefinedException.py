class TrainingDataIsNotDefinedException(Exception):
    def __init__(self):
        super().__init__("TrainingDataIsNotDefinedException")
