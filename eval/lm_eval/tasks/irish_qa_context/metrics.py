import evaluate
import string
import re
import sys
import unicodedata
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer, scoring

def process_results_gen(doc, results):
    completion = results[0]
    refs = doc['answer']
    
    bleu_scores = [bleu([[ref]], [completion]) for ref in refs]
    bleu_correct = np.nanmax(bleu_scores)

    # ROUGE-N
    rouge_scores = [rouge([ref], [completion]) for ref in refs]
    # ROUGE-1
    rouge1_scores = [score["rouge1"] for score in rouge_scores]
    rouge1_correct = np.nanmax(rouge1_scores)
    # ROUGE-2
    rouge2_scores = [score["rouge2"] for score in rouge_scores]
    rouge2_correct = np.nanmax(rouge2_scores)
    # ROUGE-L
    rougeL_scores = [score["rougeLsum"] for score in rouge_scores]
    rougeL_correct = np.nanmax(rougeL_scores)

    exact_match_metric = evaluate.load("exact_match")
    highest_exact_match = 0
    for ref in refs:
        exact_match = exact_match_metric.compute(
                        predictions=[completion], 
                        references=[ref],
                        regexes_to_ignore=["\\b(?:an |An)"],
                        ignore_case=True,
                        ignore_punctuation=True,
                        )["exact_match"]
        if exact_match > highest_exact_match:
            highest_exact_match = exact_match

    return {'bleu': bleu_correct, 
            'rouge1': rouge1_correct, 
            'rouge2': rouge2_correct, 
            'rougeL': rougeL_correct,
            'exact_match': highest_exact_match}


def bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types)
    # Add newlines between sentences to correctly compute `rougeLsum`.

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}
