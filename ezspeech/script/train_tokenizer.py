

class TextCorpus:
    def __init__(self, corpus: str, vocab: List[str] = []) -> None:
        self.corpus = corpus
        self.vocab = vocab

        self.word_freqs = self.create_word_dict(open(corpus).read().split("\n")[:-1])

    def create_new_vocab(self, vocab_size: int) -> List[int]:
        if vocab_size < len(self.vocab):
            raise Exception("New vocab size is too small")

        # Split alphabet
        vocab = self.vocab.copy()
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in vocab:
                    vocab.append(letter)

        splits = dict()
        for word in tqdm(self.word_freqs.keys()):
            spell = tokenize(word, build_vocab())
            if len(spell) > 0:
                splits[word] = spell
        while len(vocab) < vocab_size:
            scores = self.compute_pair_scores(splits)
            best_pair, max_score = "", None

            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score

            splits = self.merge_pair(*best_pair, splits)
            new_token = best_pair[0] + best_pair[1]
            if new_token not in vocab:
                print("\r" + str(len(vocab)))
                vocab.append(new_token)

        return vocab

    def create_word_dict(self, corpus: List[str]) -> Dict:
        # Create word dicts
        word_freqs = defaultdict(int)

        for text in corpus:
            text = text.lower().strip()
            words = text.split()
            for word in words:
                if word != "":
                    word_freqs[word] += 1

        return word_freqs

    def compute_pair_scores(self, splits: Dict) -> float:
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]

            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue

            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        scores = {pair: freq for pair, freq in pair_freqs.items()}
        # scores = {
        #     pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]]) ** 0.5
        #     for pair, freq in pair_freqs.items()
        # }
        return scores

    def merge_pair(self, a: str, b: str, splits: Dict) -> Dict:
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1

            splits[word] = split

        return splits