import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import gc
from .model_basic import ModelBasic
import logging

logger = logging.getLogger(__name__)


class ModelLevelChapter(ModelBasic):

    def __init__(self, database, section):
        sections_index = list(range(21))
        if section not in sections_index:
            raise Exception(f'Section {section} invalid!!!')
        self.section = section
        self.chapters_by_section = database.get_chapters_by_section(section)
        if len(self.chapters_by_section) == 1:
            self.only_one_chapter = next(iter(self.chapters_by_section))
        else:
            self.only_one_chapter = None
            if section < 10:
                section_label = '0' + str(section)
            else:
                section_label = str(section)
            model_name = f'commodity_section{section_label}_chapter_model.keras'
            super().__init__(database, model_name=model_name)
            self.label_encoder.fit(list(self.chapters_by_section.keys()))
            self.out_len = len(self.chapters_by_section.keys())

    def predict(self, description):
        if self.only_one_chapter is not None:
            return self.only_one_chapter, 1.00
        return super(ModelLevelChapter, self).predict(description)

    def get_model(self):
        if self.only_one_chapter is not None:
            logger.info(f"The section {self.section} has only one chapter. The model will not be loaded!")
            return
        super(ModelLevelChapter, self).get_model()

    def create_model(self):
        if self.only_one_chapter is not None:
            logger.info(f"The section {self.section} has only one chapter. The model will not be created!")
            return
        super(ModelLevelChapter, self).create_model()

    def train_model_in_batches(self, offset=0, batch_size=1000, epochs=10, max_records=None, fit_batch_size=32,
                               year=None, month=None):
        if self.only_one_chapter is not None:
            logger.info(f"The section {self.section} has only one chapter. The model will not be trained!")
            return
        logger.info(f"Training model offset: {offset} batch_size: {batch_size} epochs: {epochs}")
        init_offset = offset
        while True:
            if max_records is not None and offset - init_offset > max_records:
                break
            logger.info(f"Training model offset: {offset}")
            df = self.database.get_comodities(offset, batch_size, with_tariff_classification_description=False,
                                              with_chapter=True, with_section=False, year=year, month=month,
                                              section=self.section)
            if df.empty:
                break

            # Tokenizar las descripciones
            tokenized_descriptions = df['commodity_description'].apply(
                lambda x: self.tokenizer.encode(x, add_special_tokens=True))

            # Rellenar secuencias para que tengan la misma longitud

            tokenized_descriptions_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenized_descriptions,
                                                                                          padding='post')

            # Codificar las secciones
            df['commodity_chapter'] = self.label_encoder.transform(df['commodity_chapter'])
            chapters = df['commodity_chapter'].values

            # Dividir datos en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(tokenized_descriptions_padded, chapters, test_size=0.2,
                                                                random_state=42)

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

            # Entrenar el modelo
            self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=fit_batch_size,
                           callbacks=[early_stopping, reduce_lr])
            offset += batch_size
            self.model.save(self.model_path)
            logger.info(f"Model {self.model_path} saved !!!")
            del df, X_train, X_test, y_train, y_test, tokenized_descriptions, tokenized_descriptions_padded, chapters
            gc.collect()
            tf.keras.backend.clear_session()
        logger.info(f"Completed Training")
        self.model.save(self.model_path)
        logger.info(f"Model {self.model_path} saved !!!")
