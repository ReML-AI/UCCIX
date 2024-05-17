import evaluate


def bleu(predictions, references):
    return (predictions[0], references[0])


def agg_bleu(items):
    bleu_fn = evaluate.load("bleu")
    predictions, references = zip(*items)
    return bleu_fn.compute(predictions=predictions, references=references, max_order=4)["bleu"]

def ter(predictions, references):
    return (predictions[0], references[0])


def agg_ter(items):
    ter_fn = evaluate.load("ter")
    predictions, references = zip(*items)
    return ter_fn.compute(predictions=predictions, references=references, case_sensitive=True)["score"]
