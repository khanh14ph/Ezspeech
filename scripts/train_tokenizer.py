import os
import sentencepiece as spm
from tqdm import tqdm
from ezspeech.utils.common import load_jsonl

# 1. Prepare data
a = load_jsonl("/scratch/midway3/khanhnd/data/metadata/youtube_norm.jsonl")
with open("temp.txt", "w") as f:
    for i in tqdm(a):
        f.write(i["text"].strip() + "\n")

out_dir = "/scratch/midway3/khanhnd/Ezspeech/tokenizer/vi"
os.makedirs(out_dir, exist_ok=True)

# 2. Train SentencePiece
# THIS AUTOMATICALLY CREATES: tokenizer.model AND tokenizer.vocab
spm.SentencePieceTrainer.train(
    input="temp.txt",
    model_prefix=f"{out_dir}/tokenizer",
    model_type="bpe",
    user_defined_symbols=[],
    vocab_size=1024,
    input_sentence_size=10000000,
    train_extremely_large_corpus=True,
    shuffle_input_sentence=True,
    character_coverage=0.99999,
)

# 3. Generate vocab.txt to satisfy NeMo's 3rd file requirement
model_path = f"{out_dir}/tokenizer.model"
vocab_path = f"{out_dir}/vocab.txt"

sp = spm.SentencePieceProcessor(model_file=model_path)

with open(vocab_path, "w", encoding="utf-8") as f:
    for i in range(sp.get_piece_size()):
        token = sp.id_to_piece(i)
        
        # If you strictly want it to look exactly like the ## format in your image
        # keep this if/else logic. Otherwise, just do: f.write(token + "\n")
        if i >= 3: # Skip standard special tokens (unk, s, /s)
            if token.startswith(" "):
                token = token.replace(" ", "") # Remove the SentencePiece underscore
            else:
                token = "##" + token         # Add WordPiece hashes for subwords
                
        f.write(token + "\n")

# Cleanup
os.remove("temp.txt")
print(f"Success! NeMo tokenizer files generated in: {out_dir}")