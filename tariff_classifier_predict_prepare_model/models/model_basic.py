from transformers import AutoTokenizer
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import numpy as np
import logging
import config

logger = logging.getLogger(__name__)


class ModelBasic:

    def __init__(self, database, model_name, tokenizer_path=config.tokenizer_path):
        self.tokenizer = None
        self.model = None
        self.model_path = config.model_dir + '/' + model_name
        self.database = database
        self.label_encoder = LabelEncoder()
        self.out_len = None
        if not os.path.exists(tokenizer_path):
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_tokenizer_name)
            self.tokenizer.save_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def create_model(self):
        logger.info(f'Creating model: {self.model_path}')
        if os.path.exists(self.model_path):
            raise Exception(f"Modelo {self.model_path} ya existe!!")
        # Crear el modelo
        model = Sequential()
        self.model = model
        model.add(tf.keras.layers.Embedding(input_dim=self.tokenizer.vocab_size, output_dim=128))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.out_len, activation='softmax'))
        # Compilar el modelo
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def get_model(self):
        logger.info(f'Loading model: {self.model_path}')
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            logger.info("Modelo cargado desde el archivo.")
        else:
            raise Exception('Modelo no existe!!')

    def predict(self, description):
        if self.model is None:
            self.get_model()

        tokenized_desc = self.tokenizer.encode(description, add_special_tokens=True)
        tokenized_desc = tf.keras.preprocessing.sequence.pad_sequences([tokenized_desc], padding='post')
        prediction = self.model.predict(tokenized_desc)
        predicted_section = np.argmax(prediction, axis=1)
        predicted_probability = np.max(prediction, axis=1)

        return self.label_encoder.inverse_transform(predicted_section)[0], predicted_probability[0]
