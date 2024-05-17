import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_dir', default=None, type=str, required=True)
parser.add_argument('--sp_model_file', default='./spm.model', type=str)
parser.add_argument('--output_dir', default=None, type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
sp_model_file = args.sp_model_file

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
irish_sp_model = spm.SentencePieceProcessor()
irish_sp_model.Load(sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
irish_spm = sp_pb2_model.ModelProto()
irish_spm.ParseFromString(irish_sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer),len(irish_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

## Add Irish tokens to LLaMA tokenizer
llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before: {len(llama_spm_tokens_set)}")
overlap_count = 0
foreign_count = 0
for p in irish_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        FOREIGN_LETTER = re.compile(r'[^▁a-zA-ZáéíóúÁÉÍÓÚ]')
        if len(FOREIGN_LETTER.findall(piece)) == 0:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
        else:
            foreign_count += 1
    else:
        overlap_count += 1
print("NUM OVERLAP: ", overlap_count)
print("NUM FOREIGN: ", foreign_count)
print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir = f'{args.output_dir}_sp'
output_hf_dir = f'{args.output_dir}_hf' # the path to save Irish-LLaMA tokenizer
os.makedirs(output_sp_dir,exist_ok=True)
with open(output_sp_dir+'/irish_llama.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/irish_llama.model')

tokenizer.save_pretrained(output_hf_dir)
print(f"Irish-LLaMA tokenizer has been saved to {output_hf_dir}")


# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
irish_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text='''Ní ghearrann formhór na n-iar-bhunscoileanna in Éirinn táillí。
The primary use of LLaMA is research on large language models, including'''
print("Test text:\n",text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
print(f"Tokenized by Irish-LLaMA tokenizer:{irish_llama_tokenizer.tokenize(text)}")
