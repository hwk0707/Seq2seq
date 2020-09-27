import torch.nn.functional as F
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, method="concat"):
        super().__init__()

        self.method = method
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        if method == 'general':
            self.w = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        elif method == 'concat':
            self.w = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
            self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, enc_outs, dec_out):
        # enc_outs: [max_len, batch_size, encoder_dims]
        # dec_out: [1, batch_size, hid_dim]
        if self.method == 'dot':
            attn_energies = self.dot(dec_out, enc_outs)
        elif self.method == 'general':
            attn_energies = self.general(dec_out, enc_outs)
        elif self.method == 'concat':
             attn_energies = self.concat(dec_out, enc_outs)

        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1)

    def dot(self, dec_out, enc_outs):
        return torch.sum(dec_out * enc_outs, dim=2)

    def general(self, dec_out, enc_outs):
        energy = self.w(enc_outs)
        return torch.sum(dec_out * energy, dim=2)

    def concat(self, dec_out, enc_outs):

        enc_outs_len = enc_outs.shape[0]
        dec_out = dec_out.repeat(enc_outs_len, 1, 1)
        energy = self.w(torch.cat((dec_out, enc_outs), 2))
        return torch.sum(self.v * energy, dim=2)


if __name__ == "__main__":
    data = torch.rand(3, 4)
    print(data)
    print(data[:, 0].unsqueeze(1))
    a = F.softmax(data, dim=0)
    print(a)

    print(data.expand(data.shape[0], -1, -1))
    print(data.unsqueeze(1).repeat(1, 10, 1))

