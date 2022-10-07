from trandemo.data.vocabulary import Vocabulary
from trandemo.data.dataset import LineDataset, EncodedDataset, ZippedDataset, BatchedDataset, PaddedDataset, \
    TensorDataset


class DataPipeline:
    @staticmethod
    def getTrainDataset(src, tgt, src_vocab, tgt_vocab,batch_size,max_len):



        special_dict = src_vocab.get_special()

        train_dataset = LineDataset(src)
        train_dataset = EncodedDataset(train_dataset, src_vocab)
        val_dataset = LineDataset(tgt)
        val_dataset = EncodedDataset(val_dataset, tgt_vocab)

        dataset = ZippedDataset(train_dataset, val_dataset)
        dataset = BatchedDataset(dataset, batch_size)
        dataset = PaddedDataset(dataset, src_vocab.get_idx(special_dict['pad']),max_len=max_len)
        dataset = TensorDataset(dataset)
        return dataset


if __name__ == '__main__':
    # 测试pipline
    tgt = '../../data/corpus.tc.32k.zh.shuf'
    src = '../../data/corpus.tc.32k.en.shuf'
    tgt_vocab = Vocabulary('../../data/vocab.32k.en.txt', '<unk>', None, '<eos>', '<pad>')
    src_vocab = Vocabulary('../../data/vocab.32k.zh.txt', '<unk>', None, '<eos>', '<pad>')
    dataset = DataPipeline.getTrainDataset(src, tgt, src_vocab, tgt_vocab,batch_size=2,max_len=128)

    for i, data in enumerate(dataset):
        print("iter %d :" % i)
        print(data)
