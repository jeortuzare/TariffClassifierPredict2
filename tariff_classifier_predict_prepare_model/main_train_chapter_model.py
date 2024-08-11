from scripts import train_models_functions as mf
from logger_config import setup_logging

setup_logging()

if __name__ == '__main__':
    database = mf.get_database()
    for month in range(2, 5):
        for section in list(range(21)):
            mf.train_section(database, section, year=2022, month=month)
