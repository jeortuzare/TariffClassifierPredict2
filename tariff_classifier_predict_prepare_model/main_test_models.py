from scripts import train_models_functions as mf
from models.predict_classification import PredictClassification
from logger_config import setup_logging
import random
import logging

setup_logging()

logger = logging.getLogger(__name__)
test_counts = 1

if __name__ == '__main__':
    logger.info("Testing Models ...")
    database = mf.get_database()
    predict_classification = PredictClassification(database)

    rows = database.exec_query(' select min(id), max(id) from commodity', return_rows=True)
    row = rows[0]
    min_value = row[0]
    max_value = row[1]
    random_ids = [random.randint(min_value, max_value) for _ in range(test_counts)]

    query_sql = f"""select id, description as commodity_description, tariff_classification  
    FROM commodity
    WHERE id in {random_ids}""".replace('[', '(').replace(']', ')')
    rows = database.exec_query(query_sql, return_rows=True)
    bad = 0
    good = 0
    for row in rows:
        id = row[0]
        commodity_description = row[1]
        original_tariff_classification = row[2]
        logger.info(f"{original_tariff_classification} {commodity_description} ({id - min_value})")
        predicted_tariff_classification = predict_classification.predict(
            commodity_description)
        ok = original_tariff_classification == predicted_tariff_classification
        if ok:
            good += 1
        else:
            bad += 1
        logger.info(
            f"{ok} {original_tariff_classification} v/s {predicted_tariff_classification} ")
    logger.info(f"good_: {good} bad : {bad} total: {test_counts} effectivity: {good * 100 / test_counts}%")
# 9197818, 2490111
