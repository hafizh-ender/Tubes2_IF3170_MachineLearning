class InconsistenTrainingNumberOfInstances(Exception):
    def __init__(self):
        super().__init__("InconsistenTrainingNumberOfInstances")
