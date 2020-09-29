import os
project_root_path = os.path.abspath(os.path.dirname(__file__))


class DataConfig:
    def __init__(self):
        self.max_text_len = 500
        self.max_question_len = 28
        self.max_answer_len = 150
        self.batch_size = 32


class ModelConfig:
    def __init__(self):
        self.dropout_rate = 0.2
        self.embedding_dim = 300
        self.encoder_hid_dim = 300
        self.decoder_hid_dim = self.encoder_hid_dim * 2
        self.min_word_freq = 3
        self.clip = 1
        self.beam_width = 10


model_config = ModelConfig()
data_config = DataConfig()
