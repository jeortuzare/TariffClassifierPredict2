import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import gc
from .model_basic import ModelBasic
import logging

logger = logging.getLogger(__name__)


class ModelLevelPosition(ModelBasic):

    def __init__(self, database, chapter):
        self.chapter = chapter
        chapter_index = database.get_chapter_index()
        if chapter not in chapter_index:
            raise Exception(f'Chapter f{chapter} invalid!!!')
        self.positions_by_chapters = database.get_positions_by_chapter(chapter)
        model_name = f'commodity_chapter{chapter}_position_model.keras'
        super().__init__(database, model_name=model_name)
        self.label_encoder.fit(self.positions_by_chapters)
        self.out_len = len(self.positions_by_chapters)

    def train_model_in_batches(self, offset=0, batch_size=1000, epochs=10, max_records=None, fit_batch_size=32,
                               year=None, month=None):
        logger.info(f"Training model offset: {offset} batch_size: {batch_size} epochs: {epochs}")
        init_offset = offset
        while True:
            if max_records is not None and offset - init_offset > max_records:
                break
            logger.info(f"Training model offset: {offset}")
            df = self.database.get_comodities(offset, batch_size, with_tariff_classification_description=False,
                                              with_chapter=False, with_section=False, year=year, month=month,
                                              chapter=self.chapter)
            if df.empty:
                break

            # Tokenizar las descripciones
            tokenized_descriptions = df['commodity_description'].apply(
                lambda x: self.tokenizer.encode(x, add_special_tokens=True))

            # Rellenar secuencias para que tengan la misma longitud

            tokenized_descriptions_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenized_descriptions,
                                                                                          padding='post')

            # Codificar las secciones
            df['tariff_classification'] = self.label_encoder.transform(df['tariff_classification'])
            tariff_classifications = df['tariff_classification'].values

            if len(tariff_classifications) < 5:
                logger.warning(f"Very few records to split in chapter {self.chapter} offset: {offset}")
                break

            # Dividir datos en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(tokenized_descriptions_padded, tariff_classifications,
                                                                test_size=0.2, random_state=42)

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

            # Entrenar el modelo
            self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=fit_batch_size,
                           callbacks=[early_stopping, reduce_lr])
            offset += batch_size
            self.model.save(self.model_path)
            logger.info(f"Model {self.model_path} saved !!!")
            del df, X_train, X_test, y_train, y_test, tokenized_descriptions, tokenized_descriptions_padded, \
                tariff_classifications
            gc.collect()
            tf.keras.backend.clear_session()

        # Guardar el modelo y el tokenizer
        logger.info(f"Completed Training")
        self.model.save(self.model_path)
        logger.info(f"Model {self.model_path} saved !!!")
