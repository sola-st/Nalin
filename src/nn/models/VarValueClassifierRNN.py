"""

Created on 25-June-2020
@author Jibesh Patra


"""
from abc import ABC

from model import Model
import torch.nn as nn
import torch
from typing import Tuple, List, Dict


class VarValueClassifierRNN(Model, ABC):
    def __init__(self, embedding_dim: int,
                 num_of_characters_in_alphabet: int,
                 size_of_value: int,
                 model_name: str = 'VarValueClassifierRNN') -> None:
        super().__init__(embedding_dim=embedding_dim, model_name=model_name)
        self.criterion = nn.BCELoss()

        # ------- Representation for the value --------
        self.RNN_over_value = nn.GRU(
            input_size=num_of_characters_in_alphabet,
            hidden_size=embedding_dim,
            batch_first=True,
            bidirectional=True)

        self.convNetVal = nn.Sequential(
            # kernel_size --> number of characters in each 'value'
            nn.Conv1d(in_channels=num_of_characters_in_alphabet,
                      kernel_size=size_of_value,
                      out_channels=100),
            nn.ReLU(),
            nn.MaxPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # +1 because shape encoding is 101
            nn.Linear(in_features=(6 * embedding_dim) + 12, out_features=150),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=150, out_features=1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, samples: Dict) -> torch.Tensor:
        var_name_vec = samples['var'].to(self.device)

        len_vec = samples['len'].to(self.device)
        shape_vec = samples['shape'].to(self.device)
        type_vec = samples['type'].to(self.device)

        val_as_one_hot = samples['value_as_one_hot'].to(self.device)
        _, value_vec_rnn = self.RNN_over_value(val_as_one_hot)
        val_as_one_hot = val_as_one_hot.permute(0, 2, 1)
        value_vec_conv = self.convNetVal(val_as_one_hot)
        value_vec_conv = torch.squeeze(value_vec_conv)

        # Concat the hidden outputs from the layers of bi-directional GRU along with the
        # convolution over the value
        value_vec_cat = torch.cat((value_vec_rnn[0], value_vec_rnn[1], value_vec_conv), dim=1)

        concat_vec = torch.cat(
            (var_name_vec, value_vec_cat, shape_vec, len_vec, type_vec),
            dim=1)
        predictions = self.classifier(concat_vec)
        predictions_as_probabilities = self.sigmoid(predictions)
        predictions_as_probabilities = torch.squeeze(predictions_as_probabilities)
        predictions_as_probabilities = predictions_as_probabilities.double()  # The loss function expects double
        return predictions_as_probabilities

    def interpret_prediction(self, prediction: torch.Tensor, targets: torch.Tensor) -> Tuple[List, List]:
        predicted_labels = torch.round(prediction)
        predicted_labels = [int(l) for l in predicted_labels]
        targets = [int(l) for l in targets]
        return predicted_labels, targets

    def probabilities_of_being_buggy(self, prediction: torch.Tensor) -> List:
        return prediction.data.cpu().numpy()
