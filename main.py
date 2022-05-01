import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.preprocessing import Parameters
from src import LabeledTexts
from src import TextClassifier


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
        self.prepare_data()

        # Initialize the model
        self.model = TextClassifier(Parameters)

        # Training - Evaluation pipeline
        self.train()

    def prepare_data(self):

        labeled_texts = LabeledTexts(self.num_words, self.seq_len)

        # port to functional programming style with no hidden side-effects
        labeled_texts.load_data()
        labeled_texts.clean_text()
        labeled_texts.text_tokenization()
        labeled_texts.build_vocabulary()
        labeled_texts.word_to_idx()
        labeled_texts.padding_sentences()
        labeled_texts.split_data()

        self.x_train = labeled_texts.x_train
        self.y_train = labeled_texts.y_train
        self.x_test = labeled_texts.x_test
        self.y_test = labeled_texts.y_test

    def train(self):

        # Initialize dataset maper
        self.trainset_mapper = DatasetMapper(self.x_train, self.y_train)
        self.testset_mapper = DatasetMapper(self.x_test, self.y_test)

        # Initialize loaders
        self.loader_train = DataLoader(self.trainset_mapper, batch_size=self.batch_size)
        self.loader_test = DataLoader(self.testset_mapper, batch_size=self.batch_size)

        # Define optimizer
        optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

        # Starts training phase
        for epoch in range(self.epochs):
            # Set model in training model
            self.model.train()
            predictions = []
            # Starts batch training
            for x_batch, y_batch in self.loader_train:

                y_batch = y_batch.type(torch.FloatTensor)

                # Feed the model
                y_pred = self.model(x_batch)

                # Loss calculation
                loss = F.binary_cross_entropy(y_pred, y_batch)

                # Clean gradientes
                optimizer.zero_grad()

                # Gradients calculation
                loss.backward()

                # Gradients update
                optimizer.step()

                # Save predictions
                predictions += list(y_pred.detach().numpy())

            # Evaluation phase
            test_predictions = self.evaluation()

            # Metrics calculation
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
