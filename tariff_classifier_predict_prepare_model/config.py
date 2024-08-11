data_dir = '../data'
tokenizer_name = 'tokenizer.uchile'
tokenizer_path = data_dir + '/' + tokenizer_name
bert_tokenizer_name = 'dccuchile/bert-base-spanish-wwm-uncased'
model_dir =  data_dir
db_params = {
    'dbname': 'tariff_classifier_ia',
    'user': 'tariff_classifier',
    'password': 'qwerty',
    'host': 'localhost'
}