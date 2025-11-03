python ezspeech/script/infer.py \
            --checkpoint /Users/khanh/dev/asr_dev/ckpt.ckpt \
            --tokenizer /Users/khanh/dev/asr_dev/tokenizer.model \
            --input_jsonl /Users/khanh/dev/asr_dev/vietbud_test.jsonl \
            --output_file predictions.jsonl