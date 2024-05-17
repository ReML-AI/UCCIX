import sentencepiece as spm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', default=24_000, type=int)
parser.add_argument('--input', type=str)
args = parser.parse_args()

spm.SentencePieceTrainer.train(input=args.input, 
                               model_prefix='spm_10k', 
                               vocab_size=args.vocab_size, 
                               character_coverage=1.0, 
                               model_type='bpe')
