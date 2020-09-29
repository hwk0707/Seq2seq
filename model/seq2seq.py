import torch
import random
from torch import nn
from model.Encoder import Encoder
from model.Decoder import Decoder
from config import model_config


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate: hidden state
        :param previousNode: previous
        :param wordId: previous word
        :param logProb:
        :param length: length
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


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
        batch_size = answer_representation.shape[1]
        max_len = question_embed.shape[0]

        # decoder out dim is equal to vocab size
        outputs = torch.zeros(max_len, batch_size, self.vocab_size).cuda()

        # first step
        first_input = torch.tensor([bos_token] * batch_size, dtype=torch.long, device="cuda")
        input_embed = self.embedding(first_input).unsqueeze(0)
        hidden = answer_representation[-1, :, :].unsqueeze(0)

        for t in range(max_len):
            output, hidden = self.decoder(input_embed, hidden, text_representation)
            #output, hidden = self.decoder(input_embed, hidden, answer_representation)
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

    def beam_decode(self, text, answer, encoder_outputs=None):

        # text:
        beam_width = 10
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(target_tensor.size(0)):
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (
                decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
            encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([[SOS_token]], device=device)

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch
if __name__ == "__main__":
    data = torch.randn(3, 4, 5)
    print(data[:, 0].size())