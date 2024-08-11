from scripts import train_models_functions as mf

if __name__ == '__main__':
    database = mf.get_database()
    mf.train_root(database, year=2022, offset=1430000, max_records=1000000)