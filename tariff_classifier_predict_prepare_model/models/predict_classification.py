from .model_level_section import ModelLevelSection
from .model_level_chapter import ModelLevelChapter
from .model_level_position import ModelLevelPosition
import logging

logger = logging.getLogger(__name__)


class PredictClassification:

    def __init__(self, database):
        self.database = database
        self.sections_ids = database.get_sections_ids()
        self.predict_section_model = ModelLevelSection(database=database)
        self.predict_section_model.get_model()
        self.predict_chapter_models = {}
        for section in list(range(21)):
            self.predict_chapter_models[section] = None
        self.predict_position_models = {}
        for chapter in database.get_chapter_index():
            self.predict_position_models[chapter] = None

    def predict(self, description):
        section_id, section_probability = self.predict_section_model.predict(description)
        logger.info(
            f"section_id: {section_id} probability: {description} -> {self.sections_ids[section_id][0]} - {self.sections_ids[section_id][1]}")
        if self.predict_chapter_models[section_id] is None:
            self.predict_chapter_models[section_id] = ModelLevelChapter(self.database, section_id)
            self.predict_chapter_models[section_id].get_model()
        chapter_id, chapter_probability = self.predict_chapter_models[section_id].predict(description)
        if self.predict_position_models[chapter_id] is None:
            self.predict_position_models[chapter_id] = ModelLevelPosition(self.database, chapter_id)
            self.predict_position_models[chapter_id].get_model()
        position_id, position_probability = self.predict_position_models[chapter_id].predict(description)
        logger.info(f":SCP probability: {section_probability} {chapter_probability} {position_probability}")
        return position_id
