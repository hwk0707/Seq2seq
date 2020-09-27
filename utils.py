import json
import re
import itertools
from collections import Counter
from config import project_root_path


def preprocess(text):
    pass
    return text


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text_list, question_list, answer_list = [], [], []
    for d in data:
        text = d['text']
        for q_a in d['annotations']:
            text_list.append(text)
            question_list.append(q_a['Q'])
            answer_list.append(q_a['A'])
    return text_list, question_list, answer_list


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


def convert_ids_to_tokens(line, id2word):
    word_data = [id2word[l] for l in line]
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


