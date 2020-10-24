import torch
import torch.nn.functional as F
from torch import nn
from config import model_config


class Encoder(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=50, num_layer=1):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layer, bidirectional=True)
        self.dropout = nn.Dropout(model_config.dropout_rate)

        # for self-attention
        self.WS = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.update_layer = nn.Linear(4 * hidden_dim, 2 * hidden_dim, bias=False)
        self.gate = nn.Linear(4 * hidden_dim, 2 * hidden_dim, bias=False)

    def self_attn(self, outputs):

        # outputs = [max_len, batch_size, hidden_dim*2]
        # US [max_len, batch_size, hidden_dim*2]
        US = self.WS(outputs)

        # energy [batch_size, max_len, max_len]
        energy = torch.matmul(US.transpose(0, 1), outputs.transpose(0, 1).transpose(1, 2))

        # scores  [batch_size, max_len, max_len], last dim aad equal to 1
        scores = F.softmax(energy, dim=1).transpose(1, 2)

        # S [batch_size, max_len, 2*hid_dim]
        S = torch.matmul(scores, outputs.transpose(0, 1))

        # inputs [batch_size, max_len, 4*hid_dim]
        inputs = torch.cat([outputs.transpose(0, 1), S], dim=2)

        f_t = torch.tanh(self.update_layer(inputs))
        g_t = torch.sigmoid(self.gate(inputs))

        # update_output [batch_size, max_len, 2*hid_dim]
        updated_output = g_t * f_t + (1 - g_t) * outputs.transpose(0, 1)
        return updated_output.transpose(0, 1).contiguous()

    def forward(self, input_embed):

        self.rnn.flatten_parameters()
        embedded = self.dropout(input_embed)

        # outputs = [max_len, batch_size, hidden_dim*2]
        outputs, hidden = self.rnn(embedded)
        outputs = self.self_attn(outputs)
        return outputs, hidden


if __name__ == "__main__":
    import torch
    encoder = Encoder(10, 20, 2)
    input_embed = torch.randn(5, 3, 10)
    h0 = torch.randn(4, 3, 20)
    output, hn = encoder(input_embed)
    print(h0)
    print(hn, hn.size())
