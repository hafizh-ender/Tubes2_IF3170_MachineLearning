class InconsistentTrainingNumberOfInstancesException(Exception):
    def __init__(self):
        super().__init__("InconsistentTrainingNumberOfInstancesException")
