import paddle


class Vocabulary():
    def __init__(self, vocab_file, unk, bos, eos, pad):
        self._word2idx = {}
        self._idx2word = {}

        with open(vocab_file, 'rb') as f:
            for idx, word in enumerate(f):
                word = word.decode('utf-8')
                word = word.strip()
                self._word2idx[word] = idx
                self._idx2word[idx] = word

        self.special = {
            'unk': unk,
            'bos': bos,
            'eos': eos,
            'pad': pad
        }

    def __len__(self):
        return len(self._idx2word)

    def get_word(self, idx: int):
        if idx not in self._idx2word.keys():
            raise LookupError("idx : %d is not found." % idx)
        return self._idx2word[idx]

    def get_idx(self, word: str):
        if word not in self._word2idx.keys():
            raise LookupError("word : %s is not found" % word)
        return self._word2idx[word]

    def get_special(self):
        return self.special
