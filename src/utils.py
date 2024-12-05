import os

import pandas as pd


class Utils:
    @staticmethod
    def get_file_extension(filepath) -> None:
        return os.path.splitext(filepath)[1]

    @staticmethod
    def get_data_frame(data):
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame.from_records(data)