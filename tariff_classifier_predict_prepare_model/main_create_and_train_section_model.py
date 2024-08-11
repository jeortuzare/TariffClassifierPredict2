from scripts import train_models_functions as mf

if __name__ == '__main__':
    database = mf.get_database()
    mf.create_and_train_root(database, year=2022)