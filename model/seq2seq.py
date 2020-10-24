import torch
import random
from torch import nn
import operator
from queue import PriorityQueue
from model.Encoder import Encoder
from model.Decoder import Decoder
from config import model_config
from utils import init_embedding_using_w2v


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, word_id, prob, length):
        '''
        :param hiddenstate: hidden state
        :param previousNode: previous
        :param wordId: previous word
        :param logProb:
        :param length: length
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.word_id = word_id
        self.p = prob
        self.length = length

    def eval(self):
        return self.p


class Seq2seq(nn.Module):
    def __init__(self, config, vocab_size, word2id, id2word, is_load_model=False):
        super().__init__()

        self.word2id = word2id
        self.id2word = id2word
        self.vocab_size = vocab_size

        self.position_tag_embedding = nn.Embedding(len(config.tag2idx), config.position_embded_size)
        if is_load_model:
            self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        else:
            init_weight = init_embedding_using_w2v(word2id, id2word, config.w2v_path, config.embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(init_weight)
            self.embedding.weight.requires_grad = True

        self.text_encoder = Encoder(config.embedding_dim + config.position_embded_size, config.encoder_hid_dim)
        self.answer_encoder = Encoder(config.embedding_dim, config.encoder_hid_dim)

        self.decoder = Decoder(vocab_size, config.embedding_dim, config.encoder_hid_dim,
                               config.decoder_hid_dim)
        self.embedding_dropout = nn.Dropout(model_config.dropout_rate)

    def forward(self, text, answer, question, tag, teacher_forcing_ratio=0.5, is_beam_search_decode=False):

        if is_beam_search_decode:
            return self.beam_decode(text, answer, tag)

        pad_token = self.word2id['<pad>']
        bos_token = self.word2id['<start>']

        answer_embed = self.embedding(answer)
        question_embed = self.embedding(question)

        text_embed = self.embedding(text)
        tag_embed = self.position_tag_embedding(tag)
        text_with_position_embed = torch.cat((text_embed, tag_embed), dim=2)

        text_representation, _ = self.text_encoder(text_with_position_embed)
        answer_representation, _ = self.answer_encoder(answer_embed)

        # answer_representation: [max sen length, batch size, encoder dim]
        batch_size = answer_representation.shape[1]
        max_len = question_embed.shape[0]

        # decoder out dim is equal to vocab size
        outputs = torch.zeros(max_len, batch_size, self.vocab_size).cuda()

        # first step
        first_input = torch.tensor([bos_token] * batch_size, dtype=torch.long, device="cuda")
        input_embed = self.embedding(first_input).unsqueeze(0)
        # hidden = answer_representation[-1, :, :].unsqueeze(0)
        hidden = answer_representation[-1, :, :].unsqueeze(0)[-1, :, :].unsqueeze(0)

        for t in range(max_len):
            # output, hidden = self.decoder(input_embed, hidden, text_representation)
            # output, hidden = self.decoder(input_embed, hidden, answer_representation)
            output, hidden = self.decoder(input_embed, hidden, answer_representation, text_representation)
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

    def beam_decode(self, text, answer, tag):

        # text:
        beam_width = model_config.beam_width
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []

        pad_token = self.word2id['<pad>']
        bos_token = self.word2id['<start>']
        eos_token = self.word2id['<end>']

        text_embed = self.embedding(text)
        tag_embed = self.position_tag_embedding(tag)
        text_with_position_embed = torch.cat((text_embed, tag_embed), dim=2)

        answer_embed = self.embedding(answer)

        text_representations, _ = self.text_encoder(text_with_position_embed)
        answer_representations, _ = self.answer_encoder(answer_embed)

        batch_size = answer_representations.shape[1]

        # decoding goes sentence by sentence
        for idx in range(batch_size):

            text_representation = text_representations[:, idx, :].unsqueeze(1)
            answer_representation = answer_representations[:, idx, :].unsqueeze(1)

            decoder_hidden = text_representations[-1, :, :].unsqueeze(0)
            decoder_hidden = decoder_hidden[:, idx, :].unsqueeze(1)  # [max_len, 1, hid_dim]

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, p, length
            node = BeamSearchNode(decoder_hidden, None, bos_token, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000:
                    break
                # fetch the best node
                score, n = nodes.get()
                decoder_input = self.embedding(torch.tensor([n.word_id] * 1, dtype=torch.long, device="cuda")).unsqueeze(0)
                decoder_hidden = n.h

                if n.word_id == eos_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, answer_representation,
                                                              text_representation)

                # PUT HERE REAL BEAM SEARCH OF TOP
                prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].tolist()
                    p = prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.p + p, n.length + 1)
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
                utterance.append(n.word_id)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.word_id)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch, 1


if __name__ == "__main__":
    data = torch.randn(3, 4, 5)
    print(data[:, 0].size())