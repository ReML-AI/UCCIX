llama_tokenizer_dir=meta-llama/Llama-2-13b-hf
sp_model_file=spm_10k.model
output_dir=output/tok_llama2_extend_10k

python ./scripts/lang_adapt/merge_tokenizers.py \
  --llama_tokenizer_dir $llama_tokenizer_dir \
  --sp_model_file $sp_model_file \
  --output_dir $output_dir
