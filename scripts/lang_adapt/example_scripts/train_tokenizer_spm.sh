vocab_size=10000  # vocab size of new tokenizer
input=data/merged_data.txt  # input data for training tokenizer

python ./scripts/lang_adapt/train_tokenizer_spm.py \
--vocab_size $vocab_size
