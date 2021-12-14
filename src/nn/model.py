"""
Created on 21-April-2020
@author Jibesh Patra

"""
from multiprocessing.spawn import freeze_support

import torch.nn as nn
from typing import List, Tuple
import torch
from abc import ABC, abstractmethod
import time
import os
import datetime
from torch.optim.lr_scheduler import ExponentialLR

try:
    cfg = get_ipython().config
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import  tqdm
import json


class Model(nn.Module, ABC):
    r"""
    Base class for all models.

    Running training, evaluation, testing and saving of statistics is done from here.
    """

    def __init__(self, embedding_dim=0, model_name: str = 'Model'):
        super(Model, self).__init__()
        # super().__init__()
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.criterion = nn.NLLLoss()
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def interpret_prediction(self, prediction: torch.Tensor, targets: torch.Tensor) -> Tuple[List, List]:
        """
        Given, prediction and targets, interpret and return a tuple of lists
        """
        # Every model should interpret its own prediction. A general interpretation of prediction is avoided
        # since the activation function could differ between models and hence the interpretation of results.
        # Should return a Tuple of predicted labels and target labels.
        raise NotImplementedError

    @abstractmethod
    def probabilities_of_being_buggy(self, prediction: torch.Tensor) -> List:
        """
        Given the predictions, get the probability of being buggy
        :param prediction:
        :return:
        """
        raise NotImplementedError

    def run_training(self, data) -> float:
        self.train()
        running_loss = 0.
        log_interval = 2400

        for i, samples in enumerate(data):
            self.optimizer.zero_grad()
            prediction: torch.Tensor = self.__call__(
                samples)  # call the forward function
            if isinstance(self.criterion, nn.BCELoss):
                targets = samples['p_buggy'].double().to(self.device)
            else:
                targets = samples['p_buggy'].long().to(self.device)
            loss = self.criterion(prediction, targets)

            running_loss += loss.item()
            loss.backward()
            if i > 0 and (i + 1) % log_interval == 0:
                tqdm.write(
                    "Batch {:5d}/{:5d} |  t_loss {:11.2f} | ".format(i + 1, len(data), running_loss / (i + 1)))
            nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()

        avg_training_loss = round(running_loss / len(data), 2)
        return avg_training_loss

    def run_validation(self, data) -> Tuple[float, List[int], List[int]]:
        list_of_actual_labels = []
        list_of_predicted_labels = []

        self.eval()
        running_loss = 0.
        log_interval = 1200

        with torch.no_grad():
            for i, samples in enumerate(data):
                prediction: torch.Tensor = self.__call__(
                    samples)  # call the forward function

                if isinstance(self.criterion, nn.BCELoss):
                    targets = samples['p_buggy'].double().to(self.device)
                else:
                    targets = samples['p_buggy'].long().to(self.device)
                list_of_actual_labels.extend(targets)
                loss = self.criterion(
                    prediction, targets)
                running_loss += loss.item()

                pred_labels, targets = self.interpret_prediction(
                    prediction=prediction, targets=targets)
                list_of_predicted_labels.extend(pred_labels)

                if i > 0 and (i + 1) % log_interval == 0:
                    tqdm.write(
                        "Batch {:5d}/{:5d} |  v_loss {:11.2f} | ".format(i + 1, len(data), running_loss / (i + 1)))

        avg_validation_loss = round(running_loss / len(data), 2)
        return avg_validation_loss, list_of_actual_labels, list_of_predicted_labels

    def evaluation_of_prediction(self, labels: List, predicted_labels: List) -> Tuple[float, float, float, float]:
        # Evaluate p(buggy) rate
        tp, fp, tn, fn = 0, 0, 0, 0
        precision, recall, accuracy, fscore = 0., 0., 0., 0.
        for i, j in zip(labels, predicted_labels):
            pbuggy = int(i)
            pred_pbuggy = int(j)
            if pbuggy == 1:
                if pred_pbuggy == 1:
                    tp += 1
                else:
                    fn += 1
            else:  # pbuggy == 0
                if pred_pbuggy == 1:
                    fp += 1
                else:
                    tn += 1
        if tp + fp:
            precision = tp / (tp + fp)
        if tp + fn:
            recall = tp / (tp + fn)
        if tp + tn + fp + fn:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
        if precision + recall:
            fscore = (2 * precision * recall) / (precision + recall)
        return precision, recall, accuracy, fscore

    def run_testing(self, data):
        self.eval()
        predictions = []
        with torch.no_grad():
            for i, samples in tqdm(enumerate(data), desc='Making predictions'):
                prediction: torch.Tensor = self.__call__(
                    samples)  # call the forward function
                predicted_probabilities = self.probabilities_of_being_buggy(
                    prediction=prediction)
                predictions.extend(predicted_probabilities)
        return predictions

    def load_model(self, path_to_saved_model: str) -> None:
        import sys
        try:
            self.load_state_dict(
                torch.load(path_to_saved_model, map_location=self.device))
            print(f"Loaded saved model from {path_to_saved_model}")
        except FileNotFoundError:
            print(
                f"\nERROR: Pre-trained model not found at {path_to_saved_model}")
            sys.exit(1)
        except Exception as e:
            print(
                f"\nERROR: Something did not work while loading pre-trained model. Giving up {e}")
            sys.exit(1)

    def run_epochs(self, training_data, validation_data, num_epochs: int, results_dir: str):
        use_lr_scheduler = False
        if use_lr_scheduler:
            lr = 1.  # initial learning rate
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.parameters())

        best_f_score = 0.
        evaluation_scores = {'model_name': self.model_name, 'run_on': time.time(), 'precision': [], 'recall': [],
                             'accuracy': [], 'fscore': [],
                             'time_taken': []}

        for epoch in tqdm(range(num_epochs), desc='Running Epochs', ascii=' #',
                          postfix={'Model': self.model_name}):
            start_time = time.time()

            training_loss = self.run_training(data=training_data)
            validation_loss, list_of_actual_labels, list_of_predicted_labels = self.run_validation(
                data=validation_data)

            precision, recall, accuracy, fscore = self.evaluation_of_prediction(
                labels=list_of_actual_labels, predicted_labels=list_of_predicted_labels)

            evaluation_scores['precision'].append(precision)
            evaluation_scores['recall'].append(recall)
            evaluation_scores['accuracy'].append(accuracy)
            evaluation_scores['fscore'].append(fscore)
            evaluation_scores['time_taken'].append(time.time() - start_time)

            tqdm.write("-" * 110)
            tqdm.write(
                "Epoch  {:5d}/{:4d} | Training loss={:5.2f} | Validation loss={:5.2f} | lr={:3.2f} | fscore={:4.2f} | "
                "took={:4.2f} seconds".format(
                    epoch + 1,
                    num_epochs,
                    training_loss,
                    validation_loss,
                    not use_lr_scheduler or self.scheduler.get_last_lr()[0], # If the learning rate scheduler is not used, 1.0 is printed as the learning rate
                    fscore,
                    time.time() - start_time))
            tqdm.write("-" * 110)

            if use_lr_scheduler: self.scheduler.step()

            # Save the model with best f-score so far
            if round(fscore, 3) > round(best_f_score, 3):
                self.save_model(out_dir=results_dir, fscore=fscore)
                best_f_score = round(fscore, 3)

        evaluation_scores_out_file = os.path.join(results_dir, f'evaluation_results_{self.model_name}.json')
        print(f'Writing the evaluation scores to {evaluation_scores_out_file}')
        with open(evaluation_scores_out_file, 'w') as f:
            json.dump(evaluation_scores, f)

    def save_model(self, out_dir, fscore):
        current_timestamp = datetime.datetime.fromtimestamp(
            time.time()).strftime('%d-%m-%Y--%H:%M:%S')
        out_dir = os.path.join(out_dir, 'saved_models')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        saved_model_path = os.path.join(
            out_dir, f"{self.model_name}_{current_timestamp}_{str(round(fscore, 3))}.pt")
        torch.save(self.state_dict(), saved_model_path)