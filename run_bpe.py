import sentencepiece as spm

def train_sentencepiece_model(input_file, vocab_file, model_prefix='bpe', vocab_size=1024, character_coverage=1.0):
    """
    Train SentencePiece model and output tokens to file
    
    Args:
        input_file (str): Path to input text file
        vocab_file (str): Path to output vocabulary file
        model_prefix (str): Prefix for output model files
        vocab_size (int): Target vocabulary size
        character_coverage (float): Fraction of characters to include
    """
    # SentencePiece training parameters
    training_arguments = [
        f'--input={input_file}',
        f'--model_prefix={model_prefix}',
        f'--vocab_size={vocab_size}',
        f'--character_coverage={character_coverage}',
        '--model_type=bpe'
    ]
    
    # Train the model
    spm.SentencePieceTrainer.train(' '.join(training_arguments))
    
    # Load the model to extract tokens
    sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
    
    # Write tokens to file
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\n")

# Example usage
if __name__ == "__main__":
    input_file = "/home4/khanhnd/Ezspeech/ezspeech/resources/corpus.txt"  # Your input text file
    vocab_file = "bpe_tokens.txt"
    model_prefix = "bpe_model"
    
    # Train SentencePiece model and output tokens
    train_sentencepiece_model(
        input_file, 
        vocab_file,
        model_prefix=model_prefix, 
        vocab_size=2048,
        character_coverage=1.0
    )