python ezspeech/script/infer.py \
            --checkpoint /Users/khanh/dev/asr_dev/ckpt.ckpt \
            --tokenizer ezspeech/resource/tokenizer/vi/tokenizer.model \
            --input_jsonl /Users/khanh/dev/asr_dev/vietbud_test.jsonl \
            --output_file predictions.jsonl \
            --lexicon_path  ezspeech/resource/tokenizer/vi/lexicon.txt \
            --lm_path /Users/khanh/dev/asr_dev/text.arpa