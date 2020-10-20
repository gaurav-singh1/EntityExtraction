import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
EPOCHS = 10
VALID_BATCH_SIZE = 8
BASE_MODEL_PATH = "/Users/gsingh/Documents/Personnal/Projects/Bert_Sentiment_IMDB/input/bert-base-uncased"
MODEL_PATH = 'model.bin'
TRAINING_FILE = ''
TRAINING_FILE = "../input/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH, 
    do_lower_case = True
)



