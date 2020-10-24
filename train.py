import torch
from torch import optim
import numpy as np
from utils import *
from config import *
from model.seq2seq import Seq2seq
from rouge_score import rouge_scorer
import time


texts, questions, answers, text_tag_list = load_data(project_root_path + "/data/round1_train_0907.json")
test_texts, _, test_answers, test_text_tag_list = load_data(project_root_path + '/data/round1_test_0907.json')
words = texts + questions + answers + test_texts + test_answers

word2id, id2word = build_vocab(words, is_save_vocab=True, min_freq=3)
texts_id, _ = convert_tokens_to_word(texts, word2id, data_config.max_text_len)
questions_id, _ = convert_tokens_to_word(questions, word2id, data_config.max_question_len)
answers_id, _ = convert_tokens_to_word(answers, word2id, data_config.max_answer_len)
text_tag_id, _ = convert_tags_to_id(text_tag_list, data_config.max_text_len)
print(len(texts_id), len(texts_id[0]))

texts_id = torch.tensor(texts_id, dtype=torch.long)
answers_id = torch.tensor(answers_id, dtype=torch.long)
questions_id = torch.tensor(questions_id, dtype=torch.long)
text_tag_id = torch.tensor(text_tag_id, dtype=torch.long)

dataset = torch.utils.data.TensorDataset(texts_id, answers_id, questions_id, text_tag_id)

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
validation_split = 0.05
random_seed = 77
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=data_config.batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=data_config.batch_size,
                                                sampler=valid_sampler)


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0
    rouge_score_sum = 0
    count = 0

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = tuple(t.cuda() for t in batch)
            text, answer, question, tags = batch

            prediction, loss = model(text.t(), answer.t(), question.t(), tags.t(), 0) # turn off teacher forcing

            prediction = prediction.transpose(0, 1)
            prediction = prediction.max(2)[1]
            for x, y in zip(prediction, question):

                x = convert_ids_to_tokens(x.tolist())
                idx1 = x.index('<end>') if '<end>' in x else len(x)
                ans_pred = re.sub('？+', '？', ''.join(x[0:idx1]))

                q = convert_ids_to_tokens(y.tolist())
                idx2 = q.index('<pad>') if '<pad>' in q else len(q)
                q_label = ''.join(q[0:idx2])

                q_label = ' '.join([str(word2id.get(i, word2id['<unk>'])) for i in q_label])
                ans_pred = ' '.join([str(word2id.get(i, word2id['<unk>'])) for i in ans_pred])

                scores = scorer.score(q_label, ans_pred)

                rouge_score_sum += scores['rougeL'].fmeasure
                count += 1

            epoch_loss += loss.item()

    return epoch_loss / len(data_loader), rouge_score_sum/count


def train(model, data_loader, optimizer, clip, epoch):

    print_loss_total = 0
    epoch_loss = 0
    print_every = 100

    model.train()

    for i, batch in enumerate(data_loader):
        batch = tuple(t.cuda() for t in batch)
        text, answer, question, text_tag = batch
        text_t, answer_t, question_t, text_tag_t = text.t(), answer.t(), question.t(), text_tag.t()

        optimizer.zero_grad()

        prediction, loss = model(text_t, answer_t, question_t, text_tag_t, 0.5)

        print_loss_total += loss.item()
        epoch_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\tCurrent Loss: %.4f' % print_loss_avg)

        print("-------epoch {}: batch {}, loss {}-------".format(epoch, i, loss))
    return epoch_loss / len(data_loader)


if __name__ == "__main__":

    model = Seq2seq(model_config, len(word2id), word2id, id2word, True)
    model = torch.nn.DataParallel(model).cuda()
    model.module.load_state_dict(torch.load('./checkpoint/best_weight.bin'))
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    early_stop = 0
    best_loss = 1000000
    best_score = 0.434
    for epoch in range(10000000):

            train_loss = train(model, train_loader, optimizer, model_config.clip, epoch)

            valid_loss, valid_rouge_score = evaluate(model, validation_loader)
            print("rouge-L:", valid_rouge_score)

            scheduler.step()
            if valid_rouge_score > best_score:
                early_stop = 0
                best_score = valid_rouge_score
                torch.save(model.module.state_dict(), './checkpoint/best_weight.bin')
            else:
                early_stop += 1

            if early_stop >= 15:
                break

            print("======epoch:{}, best valid rouge-L score: {}======".format(epoch, best_score))


