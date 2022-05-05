from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
import re

from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from cnn.model import CNNTextClassifier

import pandas as pd
import spacy

DATA_DIR = Path(__file__).parent / 'data'


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Parameters:
    seq_len: int = 35
    vocab_size: int = 2000

    kernel_lengths: tuple = (2, 3, 4, 5)
    # Model parameters
    embedding_size: int = 64
    encoding_size: int = 32
    stride: int = 2
    strides: tuple = (2, 2, 2, 2)

    # Training parameters
    epochs: int = 10
    batch_size: int = 12
    learning_rate: float = 0.001
    test_size = 0.1


HYPERPARAMS = Parameters()


nlp = spacy.load('en_core_web_md')


def pad(sequence, pad_value=0, seq_len=HYPERPARAMS.seq_len):
    print(sequence)
    padded = list(sequence)[:seq_len]
    print(padded)
    padded = padded + [pad_value] * (seq_len - len(padded))
    return padded


def tokenize_nlp(doc):
    return [tok.text for tok in nlp(doc) if tok.text.strip()]


def tokenize_re(doc):
    return [tok for tok in re.findall(r'\w+', doc)]


def load_dataset_spacy(filepath='tweets.csv'):
    """ load and preprocess csv file: return [(token id sequences, label)...]

    1. Simplified: load the CSV
    2. NOPE: case folding:
    3. NOPE: remove non-letters (nonalpha):
    4. NOPE: remove stopwords
    5. Simplified: tokenize with regex
    6. Simplified: filter infrequent words
    7. Simplified: compute reverse index
    8. Simplified: transform token sequences to integer id sequences
    9. Simplified: pad token id sequences
    10. Simplified: train_test_split
    """

    if not Path(filepath).is_file():
        filepath = DATA_DIR / 'tweets.csv'

    # 1. Simplified: load the CSV

    df = pd.read_csv(filepath, usecols=['text', 'target'])

    # 2. NOPE: case folding:

    texts = map(str.lower, df['texts'])

    # 3. NOPE: remove non-letters (nonalpha):
    # texts = re.sub(r'[^A-Za-z]', t, ' ') for t in texts]
    # 4. NOPE: remove stopwords
    # 5. Simplified: tokenize with regex

    tokenized_texts = map(re.compile(r'\w+').findall, texts)

    # 6. Simplified: filter infrequent words

    counts = Counter(chain(*tokenized_texts))
    vocab = ['<PAD>'] + [x[0] for x in counts.most_common(HYPERPARAMS.vocab_size)]

    # 7. Simplified: compute reverse index

    tok2id = dict(zip(vocab, range(len(vocab))))

    # 8. Simplified: transform token sequences to integer id sequences

    id_sequences = [map(tok2id.get, seq) for seq in tokenized_texts]

    # 9. Simplified: pad token id sequences

    id_sequences = [list(map(pad, seq)) for seq in id_sequences]

    # 10. Simplified: train_test_split

    return dict(zip(
        'x_train x_test y_train y_test'.split(),
        train_test_split(
            X=id_sequences,
            y=df['target'],
            test_size=HYPERPARAMS.test_size,
            random_state=0)))


def load_dataset_re():
    data_filepath = Path('data') / 'tweets.csv'
    df = pd.read_csv(data_filepath.open(), usecols='text target'.split())
    texts = df['text'].values
    targets = df['target'].values
    texts = [re.sub(r'[^A-Za-z0-9.?!]+', ' ', x) for x in texts]
    texts = [tokenize_re(doc) for doc in tqdm(texts)]
    counts = Counter(chain(*texts))
    vocab = ['<PAD>'] + [x[0] for x in counts.most_common(HYPERPARAMS.vocab_size)]
    tok2id = dict(zip(vocab, range(len(vocab))))

    # 8. Simplified: transform token sequences to integer id sequences

    id_sequences = [[i for i in map(tok2id.get, seq) if i is not None] for seq in texts]

    # 9. Simplified: pad token id sequences

    padded_sequences = []
    for s in id_sequences:
        padded_sequences.append(pad(s))
    padded_sequences = torch.IntTensor(padded_sequences)

    return dict(zip(
        'x_train x_test y_train y_test'.split(),
        train_test_split(
            padded_sequences,
            targets,
            test_size=HYPERPARAMS.test_size,
            random_state=0)))


class DatasetMapper(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Controller(Parameters):

    def __init__(self):
        # self.x_train, self.y_train, self.x_test, self.y_test:
        self.__dict__.update(load_dataset_re())
        self.model = CNNTextClassifier(HYPERPARAMS)
        self.train()

    def train(self):

        self.trainset_mapper = DatasetMapper(self.x_train, self.y_train)
        self.testset_mapper = DatasetMapper(self.x_test, self.y_test)

        self.loader_train = DataLoader(self.trainset_mapper, batch_size=self.batch_size)
        self.loader_test = DataLoader(self.testset_mapper, batch_size=self.batch_size)

        optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            predictions = []
            for x_batch, y_batch in self.loader_train:
                y_batch = y_batch.type(torch.FloatTensor)
                y_pred = self.model(x_batch)
                loss = F.binary_cross_entropy(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Save predictions
                predictions += list(y_pred.detach().numpy())

            test_predictions = self.predict()
            train_accuary = self.calculate_accuracy(self.y_train, predictions)
            test_accuracy = self.calculate_accuracy(self.y_test, test_predictions)
            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch + 1, loss.item(), train_accuary, test_accuracy))

    def predict(self):

        self.model.eval()  # evaluation mode
        predictions = []

        with torch.no_grad():
            for x_batch, y_batch in self.loader_test:
                y_pred = self.model(x_batch)
                predictions += list(y_pred.detach().numpy())

        return predictions

    def calculate_accuracy(self, ground_truth, predictions):
        # Metrics calculation
        true_positives = 0
        true_negatives = 0
        for true, pred in zip(ground_truth, predictions):
            if (pred >= 0.5) and (true == 1):
                true_positives += 1
            elif (pred < 0.5) and (true == 0):
                true_negatives += 1
            else:
                pass
        # Return accuracy
        return (true_positives + true_negatives) / len(ground_truth)


if __name__ == '__main__':
    controller = Controller()
