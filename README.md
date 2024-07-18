# UCCIX: Irish-eXcellence Large Language Model
This repository hosts the codes for reproducing our Irish-based Large Language Models.

Check out our papers:
- [UCCIX: Irish-eXcellence Large Language Model](https://arxiv.org/abs/2405.13010), ECAI2024 (Demo track)
- Irish-based Large Language Model with Extreme Low-Resource Settings in Machine Translation, LoResMT 2024, ACL2024

We released our models' checkpoints and curated datasets at: https://huggingface.co/ReliableAI

## Overview
>The development of Large Language Models (LLMs) has predominantly focused on high-resource languages, leaving extremely low-resource languages like Irish with limited representation. This work presents UCCIX, a pioneering effort on the development of an open-source Irish-based LLM. We propose a novel framework for continued pre-training of LLMs specifically adapted for extremely low-resource languages, requiring only a fraction of the textual data typically needed for training LLMs according to scaling laws. Our model, based on Llama 2-13B, outperforms much larger models on Irish language tasks with up to 12% performance improvement, showcasing the effectiveness and efficiency of our approach. We also contribute comprehensive Irish benchmarking datasets, including IrishQA, a question-answering dataset, and Irish version of MT-bench. These datasets enable rigorous evaluation and facilitate future research in Irish LLM systems. Our work aims to preserve and promote the Irish language, knowledge, and culture of Ireland in the digital era while providing a framework for adapting LLMs to other indigenous languages.

> Large Language Models (LLMs) have demonstrated exceptional performances in a wide range of natural language processing tasks. However, their success does not always extend to machine translation, particularly in challenging scenarios such as translating low-resource languages. This study investigates the multilingual capability of LLMs, with a case study on Irish, an extremely low-resource language, focusing on translation tasks between English and Irish. We propose a dynamic, efficient language adaptation framework for English-centric LLMs, which involves layer-specific adjustments and subsequent fine-tuning for machine translation. Our findings highlight several key insights: (1) different layers in the LLM serve distinct functions such as language understanding and task reasoning, (2) effective translation requires extensive pre-training on both source and target languages, and (3) targeted fine-tuning for machine translation leads to significant improvements of 36.7% for English to Irish and 133.4% for Irish to English compared to the previous state-of-the-art.

## About This Implementation
Our implementations support the following features:
- Extending tokenizer's vocabulary to support languages unseen by the original tokenizer.
- Continued pre-training on the target language data, with different strategies: full-finetuning, LoRA, etc.
- Data mixture and scheduler, that allows the use of parallel data.
- Our framework for dynamic and efficient language adaptation.

## Requirements and Installation
We use *python* version 3.10, `torch==2.1.2`, `transformers==4.36.2`, and `deepspeed==0.13.3`. Other versions may caused conflicts and unstable in training.

```
pip install -r requirements.txt
```

## Tokenizer Expansion
We first train a new BPE tokenizer using `sentencepiece` on Irish data.
```
vocab_size=...  # vocabulary size of new tokenizer. By default we use a vocabulary size of 10_000
input=...  # input data for training tokenizer

python ./scripts/lang_adapt/train_tokenizer_spm.py \
        --vocab_size $vocab_size \
        --input $input
```

Then, the new tokenizer is merged with the original Llama 2's tokenizer.
```
llama_tokenizer_dir=... # the original tokenizer, e.g., meta-llama/Llama-2-13b-hf
sp_model_file=... # the newly trained tokenizer
output_dir=... # output directory to save the merged tokenizer

python ./scripts/lang_adapt/merge_tokenizers.py \
  --llama_tokenizer_dir $llama_tokenizer_dir \
  --sp_model_file $sp_model_file \
  --output_dir $output_dir
```

## Language Adaptation
Run `run_clm_pt_with_peft.py` to finetune language model on a new language. 
- `pretrained_model`: original model.
- `irish_tokenizer_path`: path directory to the expanded tokenizer (or use the same string as `pretrained_model` for the original tokenizer).
- `dataset_dir`: path directory to the dataset.
- `trainer`: set as "custom" to use proposed data scheduler in training, or set to "default" to use the default HuggingFace Trainer.
```
lr=1e-4
pretrained_model=...
irish_tokenizer_path=...
dataset_dir=...
trainer=...
data_cache=cache
output_dir=output_model
per_device_train_batch_size=2
per_device_eval_batch_size=1
gradient_accumulation_steps=8

deepspeed_config_file=./scripts/lang_adapt/example_scripts/ds_zero3_no_offload.json

torchrun --nnodes 1 --nproc_per_node 6 --master_port 29502 ./scripts/lang_adapt/run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${irish_tokenizer_path} \
    --trainer ${trainer} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed 24259 \
    --fp16 \
    --num_train_epochs 4 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 100 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 4096 \
    --output_dir ${output_dir} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --overwrite_output_dir \
    --use_peft False

```

To perform continued pre-training with our dynamic adaptation framework, you can set `langdeplay=Default`, or `langdeplay=Revert` for the ablation study with fine-tuning reasoning layers.

Additionally, you can also set `use_peft=True` to train with parameter efficient fine-tuning techniques, such as LoRA, (IA)^3, etc. (used in preliminary experiments).

## Evaluation
We adopt the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework by EleutherAI for evaluating models on Irish language tasks.

### Installation
We recommend creating a new virtual environment to avoid conflicts with libraries used during training.
```
cd eval
pip install -e .
```

### Evaluation on Irish Language Tasks
The raw data are available in `lm_eval/tasks/{irish_cloze,irish_qa_context,irish_sib200,irish_gaHealth}`.

Please reference the original framework for more customization.
The following scripts can be used to run evaluation on all Irish tasks:
```
m=... # path to model

# IrishQA (Ours):
lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks irish_qas_context \
        --num_fewshot 5 --output_path output/temp --log_samples \
        --device cuda:0 --wandb_args project=irish_llm_evaluation

# SIB200 (Irish subset):
lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks irish_sib200 \
        --output_path output/temp --log_samples \
        --device cuda:0 --wandb_args project=irish_llm_evaluation --num_fewshot 10

# Cloze Test:
lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks irish_cloze \
        --num_fewshot 0 --log_samples \
        --device cuda:0 --output_path output/temp --wandb_args project=irish_llm_evaluation

# LoResMT:
lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks gaHealth \
        --num_fewshot 5 --output_path output/temp \
        --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation
```

## Citation
TBU
