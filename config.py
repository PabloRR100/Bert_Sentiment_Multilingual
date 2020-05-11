
import transformers

TPUs = False
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

LR = 3e-5
EPOCHS = 2 # 10
MODEL_PATH = 'pytorch_model.bin'
# TRAINING_FILE = "../input/imdb.csv"
BERT_PATH = 'bert-base-multilingual-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)