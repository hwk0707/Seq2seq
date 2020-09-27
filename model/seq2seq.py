import torch
import random
from torch import nn
from model.Encoder import Encoder
from model.Decoder import Decoder
from config import model_config


class Seq2seq(nn.Module):
    def __init__(self, config, vocab_size, word2id, id2word):
        super().__init__()

        self.word2id = word2id
        self.id2word = id2word
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)

        self.text_encoder = Encoder(config.embedding_dim, config.encoder_hid_dim)
        self.answer_encoder = Encoder(config.embedding_dim, config.encoder_hid_dim)

        self.decoder = Decoder(vocab_size, config.embedding_dim, config.encoder_hid_dim,
                               config.decoder_hid_dim)
        self.embedding_dropout = nn.Dropout(model_config.dropout_rate)

    def forward(self, text, answer, question, teacher_forcing_ratio=0.5):

        pad_token = self.word2id['<pad>']
        bos_token = self.word2id['<start>']

        text_embed = self.embedding(text)
        answer_embed = self.embedding(answer)
        question_embed = self.embedding(question)

        text_representation, _ = self.text_encoder(text_embed)
        answer_representation, _ = self.answer_encoder(answer_embed)

        # answer_representation: [max sen length, batch size, encoder dim]
        batch_size = text_representation.shape[1]
        max_len = question_embed.shape[0]

        # decoder out dim is equal to vocab size
        outputs = torch.zeros(max_len, batch_size, self.vocab_size).cuda()

        # first step
        first_input = torch.tensor([bos_token] * batch_size, dtype=torch.long, device="cuda")
        input_embed = self.embedding(first_input).unsqueeze(0)
        hidden = answer_representation[-1, :, :].unsqueeze(0)

        for t in range(max_len):
            output, hidden = self.decoder(input_embed, hidden, text_representation)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top = output.max(1)[1]
            top_embed = self.embedding(top).unsqueeze(0)
            input_embed = (question_embed[t].unsqueeze(0) if teacher_force else top_embed)

        loss_fn = nn.NLLLoss(ignore_index=pad_token)
        loss = loss_fn(
            outputs.reshape(-1, len(self.word2id)),  # [batch*seq_len, output_dim]
            question.reshape(-1)  # [batch*seq_len]
        )
        return outputs, loss


if __name__ == "__main__":
    data = torch.randn(3, 4, 5)
    print(data[:, 0].size())