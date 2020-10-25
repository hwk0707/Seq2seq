import json
import re
import itertools
from collections import Counter
from config import project_root_path
import gensim
import torch
import re
from config import model_config


def preprocess(text):
    pass
    return text


def init_embedding_using_w2v(word2idx, idx2word, path, embed_size):
    w2v = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    weight = torch.zeros(len(word2idx), embed_size)

    for i in range(len(idx2word)):
        word = idx2word[i]
        index = word2idx[word]
        try:
            weight[index, :] = torch.from_numpy(w2v.get_vector(word))
        except:
            weight[index, :] = torch.rand(1, embed_size)
            print("{} not in the w2v vocab!".format(word))
    return weight


def convert_tokens_to_word(data, vocab, max_len, if_go=False):
    processed_data = []
    length_data = []
    for line in data:
        encode = []
        if if_go:
            encode.append(vocab['<start>'])
        for word in line:
            encode.append(vocab[word] if word in vocab else vocab['<unk>'])
        if if_go:
            encode.append(vocab['<end>'])
        length_data.append(len(encode))
        encode = encode + [vocab['<pad>']] * (max_len - len(encode))
        processed_data.append(encode[:max_len])
    return processed_data, length_data


def load_data(path, word2id={}):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # oov list for point copy net
    text_list, text_tag_list, question_list, answer_list, oov_list = [], [], [], [], []
    for d in data:
        text = d['text']
        # print("text", text)
        for q_a in d['annotations']:
            text_list.append(text)

            if not word2id:
                oov_list.append([str(word) for word in text if str(word) not in word2id])
            else:
                oov_list.append([0])

            q_a['A'] = q_a['A'].strip().strip(" ")
            question_list.append(q_a['Q'])
            answer_list.append(q_a['A'])

            ans_split_list = re.findall("[^，。？]+[，。？]?", q_a['A'])
            position_embed = ["O"] * len(text)
            for idx, ans in enumerate(ans_split_list):
                ans_start = -1
                if ans in text:
                    ans_start = text.index(ans)
                elif ans[0:-4] in text:
                    ans_start = text.index(ans[0:-4])
                elif len(ans) > 4 and ans[4:] in text:
                    ans_start = text.index(ans[4:])

                if ans_start != -1:
                    ans_end = ans_start + len(ans)
                    position_embed[ans_start:ans_end] = ["I"] * (ans_end - ans_start)
                if idx == len(ans_split_list) - 1:
                    start = position_embed.index("I")
                    position_embed[start] = "B"
            # try:
            #     print(q_a['A'], ''.join(position_embed).index("B"), text.index(q_a['A']),list(''.join(position_embed)).count("B"), list(''.join(position_embed)).count("I"),len(q_a['A'],))
            # except:
            #     try:
            #         print("======", q_a['A'], ''.join(position_embed).index("B"), "=====")
            #     except Exception as e:
            #         print("======", q_a['A'], ''.join(position_embed).index('I'), "=====!!!!!!")
            #         raise e

            text_tag_list.append(''.join(position_embed))

    return text_list, question_list, answer_list,  text_tag_list, oov_list


def build_vocab(words_list, is_save_vocab=False, min_freq=3):

    words_list = itertools.chain.from_iterable(words_list)
    counter = Counter(words_list)
    word2idx = {'<pad>': 0,
                '<start>': 1,
                '<end>': 2,
                '<unk>': 3}
    idx2word = dict()
    for word, idx in word2idx.items():
        idx2word[idx] = word
    idx = 4
    for word, value in counter.items():
        if value >= min_freq:
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1
    if is_save_vocab:
        with open(project_root_path + "/checkpoint/word2idx.json", 'w') as f:
            json.dump(word2idx, f)
        with open(project_root_path + "/checkpoint/idx2word.json", 'w') as f:
            json.dump(idx2word, f)
    return word2idx, idx2word


def load_vocab():
    with open(project_root_path + "/checkpoint/word2idx.json", 'r') as f:
        word2idx = json.load(f)
    with open(project_root_path + "/checkpoint/idx2word.json", 'r') as f:
        id2word = json.load(f)
    return word2idx, id2word


def convert_tokens_to_word(data, vocab, max_len, if_go=False, oov=""):
    processed_data = []
    length_data = []
    for line in data:
        encode = []
        if if_go:
            encode.append(vocab['<start>'])
        for word in line:
            if word in oov:
                encode.append(int(oov.index(word)) + len(vocab))
            else:
                encode.append(vocab[word] if word in vocab else vocab['<unk>'])
        if if_go:
            encode.append(vocab['<end>'])
        length_data.append(len(encode))
        encode = encode + [vocab['<pad>']] * (max_len - len(encode))
        processed_data.append(encode[:max_len])
    return processed_data, length_data


def convert_tags_to_id(data, max_len):
    processed_data = []
    length_data = []
    for line in data:
        encode = []
        for word in line:
            encode.append(model_config.tag2idx[word])
        encode = encode + [model_config.tag2idx['PAD']] * (max_len - len(encode))
        processed_data.append(encode[:max_len])
    return processed_data, length_data


def convert_ids_to_tokens(line, oov=""):
    word2id, id2word = load_vocab()
    if not oov:
        word_data = []
        for l in line:
            if str(l) in id2word:
                word_data.append(id2word[str(l)])
            else:
                word_position = max(int(l) - len(word2id), 0)
                word_data.append(oov[word_position])
    else:
        word_data = [id2word[str(l)] for l in line]
    return word_data


def calculate_rouge(prediction, ground_truth, id2word, nlgeval):
    prediction = prediction.max(2)[1]
    references = []
    hypotheses = []
    for x, y in zip(ground_truth, prediction):
        x = convert_ids_to_tokens(x.tolist(), id2word)
        y = convert_ids_to_tokens(y.tolist(), id2word)
        idx1 = x.index('<end>') if '<end>' in x else len(x)
        idx2 = y.index('<end>') if '<end>' in y else len(y)
        x = re.sub('\n', '', ' '.join(x[1:idx1]))
        y = re.sub('\n', '', ' '.join(y[1:idx2]))
        references.append([x])
        hypotheses.append(y)

    metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    return metrics_dict['ROUGE_L'], references, hypotheses


if __name__ == "__main__":
    pass


