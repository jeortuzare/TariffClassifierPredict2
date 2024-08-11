from scripts import train_models_functions as mf
from logger_config import setup_logging

setup_logging()

if __name__ == '__main__':
    database = mf.get_database()
    for month in range(1, 2):
        for section in list(range(21)):
            mf.create_and_train_section(database, section, year=2022, month=month)
