import sentencepiece as spm
tokenizer_version="vi"
sp = spm.SentencePieceProcessor(model_file=f'ezspeech/resource/tokenizer/{tokenizer_version}/tokenizer.model')
vn_words=open("ezspeech/resource/vn_word_lst.txt").read().splitlines()

with open(f"/Users/khanh/dev/Ezspeech/ezspeech/resource/tokenizer/{tokenizer_version}/lexicon.txt","w") as f:
    for i in vn_words:
        res=sp.encode(i)
        if 0 in res:
            continue
        res=sp.encode(i, out_type=str)
        f.write(i+"\t"+" ".join(res)+"\n")

