from functools import partial
from multiprocessing import Pool

from ezspeech.moduels.dataset.utils.text import tokenize
from ezspeech.utils.common import load_dataset, save_dataset


def process_item(item, vocab):
    """Process a single item with tokenization"""
    item["tokenized_transcript"] = tokenize(item["transcript"], vocab)
    return item


def main():
    # Load vocabulary
    vocab = open("ezspeech/resource/vocab/vi_en.txt").read().splitlines()
    old_path = "huhu"
    new_path = "hehe"

    # Load dataset
    data = load_dataset(old_path)

    # Create a partial function with vocab already set
    process_func = partial(process_item, vocab=vocab)

    # Use multiprocessing pool
    with Pool() as pool:
        # Map the processing function to all items
        processed_data = pool.map(process_func, data)

    # Save the processed dataset
    save_dataset(processed_data, new_path)


if __name__ == "__main__":
    main()
