from scripts import train_models_functions as mf
from logger_config import setup_logging

setup_logging()

if __name__ == '__main__':
    database = mf.get_database()
    for chapter in database.get_chapter_index(2):
        mf.train_chapter(database, chapter, year=2022, epochs=100)
