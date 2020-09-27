from torch import nn
from config import model_config


class Encoder(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=50, num_layer=1):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layer, bidirectional=True)
        self.dropout = nn.Dropout(model_config.dropout_rate)

    def forward(self, input_embed):

        self.rnn.flatten_parameters()
        embedded = self.dropout(input_embed)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


if __name__ == "__main__":
    import torch
    encoder = Encoder(10, 20, 2)
    input_embed = torch.randn(5, 3, 10)
    h0 = torch.randn(4, 3, 20)
    output, hn = encoder(input_embed)
    print(h0)
    print(hn, hn.size())
