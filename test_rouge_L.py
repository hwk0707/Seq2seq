from rouge_score import rouge_scorer
from rouge import Rouge

if __name__ == "__main__":
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score('1 2 3 4 5',
                          '1 2 3 4 5 8')
    print(scores['rougeL'].fmeasure)

    hypothesis = u"你好"

    reference = u"你好吗kjdjkdkj"

    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    print(scores[0]["rouge-l"]['f'])