from models.model_level_section import ModelLevelSection
from models.model_level_chapter import ModelLevelChapter
from models.model_level_position import ModelLevelPosition
from database.database import Database
import config


def get_database():
    return Database(**config.db_params)


def test_section(database):
    predictor = ModelLevelSection(database)
    test_cases_count = 1000
    test_cases_offset = 1000000

    comodities = database.get_comodities(test_cases_offset, test_cases_count,
                                         with_tariff_classification_description=False,
                                         with_chapter=False, with_section=True)
    bad = 0
    good = 0
    for index, row in comodities.iterrows():
        predicted_section, predicted_section_probability = predictor.predict(row['commodity_description'])
        ok = predicted_section == row['commodity_section']
        if ok:
            good += 1
        else:
            bad += 1
        print(ok, row['commodity_description'], row['commodity_section'], predicted_section,
              predicted_section_probability)
    print('good:', good, int(good * 100 / test_cases_count), '%')
    print('bad:', bad, int(bad * 100 / test_cases_count), '%')


def create_and_train_root(database, year=None, month=None):
    predictor = ModelLevelSection(database)
    predictor.create_model()
    predictor.train_model_in_batches(batch_size=10000, epochs=50, year=year, month=month)


def train_root(database, year=None, month=None, offset=0, max_records=None):
    predictor = ModelLevelSection(database)
    predictor.get_model()
    predictor.train_model_in_batches(batch_size=10000, epochs=50, year=year, month=month, offset=offset,
                                     max_records=max_records)


def create_and_train_section(database, section, year=None, month=None):
    predictor = ModelLevelChapter(database, section)
    predictor.create_model()
    predictor.train_model_in_batches(year=year, month=month, batch_size=10000, epochs=50)


def train_section(database, section, year=None, month=None):
    predictor = ModelLevelChapter(database, section)
    predictor.get_model()
    predictor.train_model_in_batches(year=year, month=month, batch_size=10000, epochs=50)


def create_and_train_chapter(database, chapter, year=None, month=None, epochs=50):
    predictor = ModelLevelPosition(database, chapter)
    predictor.create_model()
    predictor.train_model_in_batches(year=year, month=month, batch_size=10000, epochs=epochs)


def train_chapter(database, chapter, year=None, month=None, epochs=50):
    predictor = ModelLevelPosition(database, chapter)
    predictor.get_model()
    predictor.train_model_in_batches(year=year, month=month, batch_size=10000, epochs=epochs)

# Evaluar el modelo
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Loss: {loss}, Accuracy: {accuracy}')
