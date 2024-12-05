class NotFittedException(Exception):
    def __init__(self, message="NotFittedException"):
        super().__init__(message)