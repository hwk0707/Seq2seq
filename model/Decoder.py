import torch
from torch import nn
from model.Attention import Attention
from config import model_config, data_config


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()

        # self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(model_config.dropout_rate)
        self.rnn = nn.GRU(emb_dim, dec_hid_dim)
        self.attention = Attention(enc_hid_dim * 2, dec_hid_dim)
        self.out = nn.Linear(dec_hid_dim, output_dim)
        self.concat = nn.Linear(dec_hid_dim * 2, dec_hid_dim)

        self.ans_text_att = nn.Linear(data_config.max_text_len, enc_hid_dim * 2)
        self.embedding_dropout = nn.Dropout(model_config.dropout_rate)
        self.soft_max = nn.LogSoftmax(dim=1)

    def ans_text_attention(self, ans_enc_outs, text_en_outs):

        # tranform: [batch_size, enc_hid_dim, enc_hid_dim*2]
        # ans_enc_outs: [max_sen_len, batch_size, enc_hid_dim*2]
        trans = self.ans_text_att(text_en_outs.transpose(0, 1).transpose(1, 2))

        ans_enc_outs = ans_enc_outs.transpose(0, 1)  # [batch_size, max_sen_len, enc_hid_dim*2]
        ans_text_att_outs = torch.bmm(ans_enc_outs, trans)  # [batch_size, max_sen_len, enc_hid_dim*2]
        ans_text_att_outs = ans_text_att_outs.transpose(0, 1)  # [max_sen_len, batch_size, enc_hid_dim*2]

        return ans_text_att_outs

    def forward(self, input_embed, hidden, ans_enc_outs, text_enc_outs):

        # input_embed:[1, batch_size, emb_dim]
        # ans_enc_outs: [max_anwser_len, batch_size, enc_hid_dim]
        # text_enc_outs:
        # hidden: [1, batch_size, decoder_hid_dim]
        self.rnn.flatten_parameters()

        dec_out, hidden = self.rnn(input_embed, hidden)

        #att_enc_outs = self.ans_text_attention(ans_enc_outs, text_enc_outs)

        # att_w: [batch_size, 1, max_len]
        #att_w = self.attention(att_enc_outs, hidden).unsqueeze(1)
        att_w = self.attention(text_enc_outs, hidden).unsqueeze(1)
        # att_w = self.attention(ans_enc_outs, hidden).unsqueeze(1)

        # weighted_context: [batch_size, 1, dec_hid_dim]
        #weighted_context = torch.bmm(att_w, att_enc_outs.transpose(0, 1))
        weighted_context = torch.bmm(att_w, text_enc_outs.transpose(0, 1))
        # weighted_context = torch.bmm(att_w, ans_enc_outs.transpose(0, 1))

        # [batch_size, dec_hid_dim * 2
        concat_context = torch.cat((weighted_context, dec_out.transpose(0, 1)), dim=2).squeeze(1)
        concat_out = torch.tanh(self.concat(concat_context))  # [batch_size, dec_out_dim]
        prediction = self.out(concat_out)  # [batch_size, output_dim]
        prediction = self.soft_max(prediction)
        return prediction, hidden


if __name__ == "__main__":
    import torch
    input_tensor = torch.randn(5, 4, 10)
    a = nn.Linear(10, 12)
    print(a(input_tensor).shape)
    transform = torch.randn(10, 4).repeat(5, 1, 1)

    print(torch.bmm(input_tensor, transform).shape)
    print(input_tensor.size())
    input_tensor = input_tensor.unsqueeze(0)
    print(input_tensor.size())

    rnn = nn.GRU(10, 20)
    input_data = torch.randn(1, 3, 10)
    h0 = torch.randn(1, 3, 20)
    output, hn = rnn(input_data, h0)
    print(hn.size())