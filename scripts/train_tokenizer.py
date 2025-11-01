import sentencepiece as spm

spm.SentencePieceTrainer.train(input='/home3/khanhnd/youtube_corpus.txt', model_prefix='m', vocab_size=1024,character_coverage=1.0,model_type="bpe",bos_id=-1,eos_id=-1,pad_id=-1)
