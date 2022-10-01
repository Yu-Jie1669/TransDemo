from typing import Iterator

import paddle

from vocabulary import Vocabulary


class IterBase(Iterator):
    def __init__(self, dataset):
        self._dataset = dataset

    def __next__(self):
        return next(self._dataset)

    def reset(self):
        raise NotImplementedError


class LineIter(IterBase):
    def __init__(self, filename, dataset):
        super(LineIter, self).__init__(dataset)
        self._filename = filename
        self._file = open(filename, 'rb')

    def __next__(self):
        line = next(self._file)
        return line.decode('utf-8')

    def reset(self):
        self._file.seek(0)


class EncodedIter(IterBase):
    def __init__(self, dataset, vocab: Vocabulary):
        super(EncodedIter, self).__init__(dataset)
        self._vocab = vocab
        self._special = vocab.get_special()

    def __next__(self):
        line = next(self._dataset)
        tokens = line.strip().split(' ')

        def word2idx_func(x):
            try:
                idx = self._vocab.get_idx(x)
                return idx
            except LookupError:
                unk = self._special['unk']
                idx = self._vocab.get_idx(unk)
                return idx

        idxs = [word2idx_func(x) for x in tokens]
        if self._special['bos']:
            bos_idx = self._vocab.get_idx(self._special['bos'])
            idxs.insert(0, bos_idx)
        if self._special['eos']:
            eos_idx = self._vocab.get_idx(self._special['eos'])
            idxs.append(eos_idx)
        return idxs, len(idxs)

    def reset(self):
        self._dataset.reset()


class ZippedIter(IterBase):
    def __init__(self, datasets):
        super(ZippedIter, self).__init__(None)
        self._datasets = datasets

    def __next__(self):
        data = []
        lens = []
        for dataset in self._datasets:
            idxs, len_ = next(dataset)
            data.append(idxs)
            lens.append(len_)
        return tuple(data), tuple(lens)

    def reset(self):
        for dataset in self._datasets:
            dataset.reset()


class BatchedIter(IterBase):
    def __init__(self, dataset, batch_size):
        super(BatchedIter, self).__init__(dataset)
        self._batch_size = batch_size

    def __next__(self):
        batch_data = []
        lens = []
        for i in range(self._batch_size):
            try:
                data, len_ = next(self._dataset)
                batch_data.append(data)
                lens.append(len_)
            except StopIteration as e:
                if len(batch_data):
                    break
                else:
                    self.reset()
                    raise e
        return batch_data, lens

    def reset(self):
        self._dataset.reset()


class PaddedIter(IterBase):
    def __init__(self, dataset, pad_idx, pad_nums=None):
        super(PaddedIter, self).__init__(dataset)
        self._pad_idx = pad_idx
        self._pad_nums = pad_nums

    def __next__(self):
        batch_data, lens = next(self._dataset)

        if self._pad_nums is None:
            self._pad_nums = len(lens[0])

        max_lens = []
        for i in range(self._pad_nums):
            max_len = max([len_[i] for len_ in lens])
            max_lens.append(max_len)

        for data in batch_data:
            for index in range(self._pad_nums):
                while len(data[index]) < max_lens[index]:
                    data[index].append(self._pad_idx)

        input_data = [[] for _ in range(len(lens[0]))]
        for data in batch_data:
            for index in range(len(lens[0])):
                input_data[index].append(data[index])

        return tuple(input_data)


class TensorIter(IterBase):
    def __init__(self, dataset):
        super(TensorIter, self).__init__(dataset)

    def __next__(self):
        batch_data = next(self._dataset)
        batch_data = tuple([paddle.to_tensor(data) for data in batch_data])
        return batch_data
