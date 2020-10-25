import torch
from torch import optim
import numpy as np
import re
from utils import *
from config import *
from model.seq2seq import Seq2seq


word2id, id2word = load_vocab()

test_texts, _, test_answers, text_tag_list, oov_list = load_data(project_root_path + '/data/juesai_1011.json', word2id)

test_texts_id, _ = convert_tokens_to_word(test_texts, word2id, data_config.max_text_len)
test_answers_id, _ = convert_tokens_to_word(test_answers, word2id, data_config.max_answer_len)
text_tag_id, _ = convert_tags_to_id(text_tag_list, data_config.max_text_len)

test_texts_id = torch.tensor(test_texts_id, dtype=torch.long)
test_answers_id = torch.tensor(test_answers_id, dtype=torch.long)
text_tag_id = torch.tensor(text_tag_id, dtype=torch.long)

test_dataset = torch.utils.data.TensorDataset(test_texts_id, test_answers_id, text_tag_id)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=data_config.batch_size, shuffle=False)


def greedy_decode():
    model = Seq2seq(model_config, len(word2id), word2id, id2word, True)
    model = torch.nn.DataParallel(model).cuda()
    model.module.load_state_dict(torch.load('./checkpoint/best_weight.bin'))

    model.eval()
    test_question = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = tuple(t.cuda() for t in batch)
            text, answer, tag = batch
            bs = text.size(0)
            q = torch.ones(bs, data_config.max_question_len).long().cuda()
            prediction, _ = model(text.t(), answer.t(), q.t(), tag.t(),0)
            prediction = prediction.transpose(0, 1)
            prediction = prediction.max(2)[1]
            for x in prediction:
                x = convert_ids_to_tokens(x.tolist())
                idx1 = x.index('<end>') if '<end>' in x else len(x)
                test_question.append(''.join(x[0:idx1]))
                print(''.join(x[1:idx1]), "\n")

    with open(project_root_path + '/data/juesai_1011.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    index = 0
    submit = []
    for line in data:
        new_line = {}
        new_annotations = []
        for annotation in line['annotations']:
            annotation['Q'] = re.sub('？+', '？', test_question[index])
            index += 1
            new_annotations.append(annotation)
        new_line['id'] = line['id']
        new_line['text'] = line['text']
        new_line['annotations'] = new_annotations
        submit.append(new_line)

    with open('./prediction_result/result.json', 'w', encoding='utf-8') as f:
        json.dump(submit, f, indent=4, ensure_ascii=False)


def beam_search_decode():

    model = Seq2seq(model_config, len(word2id), word2id, id2word, True)
    model = torch.nn.DataParallel(model).cuda()
    model.module.load_state_dict(torch.load('./checkpoint/best_weight.bin'))

    model.eval()
    test_question = []
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = tuple(t.cuda() for t in batch)
            text, answer, tags = batch
            bs = text.size(0)
            q = torch.ones(bs, data_config.max_question_len).long().cuda()
            try:
                prediction, _ = model(text.t(), answer.t(), q.t(), tags.t(), 0, True)
                for x in prediction:
                    q_id_list = x[0]
                    q_text_list = convert_ids_to_tokens(q_id_list)
                    idx1 = q_text_list.index('<end>') if '<end>' in q_text_list else len(q_text_list)
                    test_question.append(''.join(q_text_list[0:idx1]))
                    print(''.join(q_text_list[0:idx1]) + "？", "\n")
                print("==========================================")
            except:
                count += 1
                print("beam_search_failed count is {}".format(str(count)))
                prediction, _ = model(text.t(), answer.t(), q.t(), tags.t(), 0)
                prediction = prediction.transpose(0, 1)
                prediction = prediction.max(2)[1]
                for x in prediction:
                    x = convert_ids_to_tokens(x.tolist())
                    idx1 = x.index('<end>') if '<end>' in x else len(x)
                    test_question.append(''.join(x[0:idx1]))
                    print(''.join(x[0:idx1]), "\n")

    with open(project_root_path + '/data/juesai_1011.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    index = 0
    submit = []
    for line in data:
        new_line = {}
        new_annotations = []
        for annotation in line['annotations']:
            annotation['Q'] = re.sub('？+', '？', test_question[index])
            index += 1
            new_annotations.append(annotation)
        new_line['id'] = line['id']
        new_line['text'] = line['text']
        new_line['annotations'] = new_annotations
        submit.append(new_line)

    with open('./prediction_result/result.json', 'w', encoding='utf-8') as f:
        json.dump(submit, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
   greedy_decode()
   # beam_search_decode()