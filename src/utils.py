import os


class Utils:
    @staticmethod
    def get_file_extension(filepath) -> None:
        return os.path.splitext(filepath)[1]
