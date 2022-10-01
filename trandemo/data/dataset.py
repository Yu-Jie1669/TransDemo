import paddle
from paddle.io import IterableDataset

# from  dataset.iterator import IterBase, LineIter, EncodedIter
from trandemo.data.iterator import IterBase, LineIter, EncodedIter, ZippedIter, BatchedIter, PaddedIter, TensorIter


class DatasetBase(IterableDataset):
    def __init__(self):
        super(DatasetBase, self).__init__()
        self.iter = IterBase

    def __iter__(self):
        return self.iter

    def __next__(self):
        return next(self.iter)

    def reset(self):
        self.iter.reset()


class LineDataset(DatasetBase):

    def __init__(self, filename):
        super(LineDataset, self).__init__()
        self.iter = LineIter(filename, self)


class EncodedDataset(DatasetBase):
    def __init__(self, dataset: LineDataset, vocab):
        super(EncodedDataset, self).__init__()
        self._dataset = dataset
        self.vocab = vocab
        self.iter = EncodedIter(dataset, vocab)

    def get_vocab(self):
        return self.vocab


class ZippedDataset(DatasetBase):
    def __init__(self, *datasets):
        super(ZippedDataset, self).__init__()
        self._datasets = datasets
        self.iter = ZippedIter(datasets)


class BatchedDataset(DatasetBase):
    def __init__(self, dataset, batch_size):
        super(BatchedDataset, self).__init__()
        self._batch_size = batch_size
        self._dataset = dataset
        self.iter = BatchedIter(dataset, batch_size)


class PaddedDataset(DatasetBase):
    def __init__(self, dataset,pad_idx):
        super(PaddedDataset, self).__init__()
        self.iter=PaddedIter(dataset,pad_idx)

class TensorDataset(DatasetBase):
    def __init__(self,dataset):
        super(TensorDataset, self).__init__()
        self.iter=TensorIter(dataset)


