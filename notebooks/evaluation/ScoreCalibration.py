import torch
import torch.nn as nn
import numpy as np


def cllr(target_llrs, nontarget_llrs):
    def neg_log_sigmoid(lodds):
        return torch.log1p(torch.exp(-lodds))

    return 0.5 * (
        torch.mean(neg_log_sigmoid(target_llrs)) +
        torch.mean(neg_log_sigmoid(-nontarget_llrs))
    ) / np.log(2)


class ScoreCalibrationModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.W = nn.Linear(input_dim, 1)

        nn.init.constant_(self.W.weight, 1.0 / input_dim)
        nn.init.constant_(self.W.bias, 0)
    
    def forward(self, X):
        return self.W(X)


class ScoreCalibration:

    def __init__(self, evaluations):
        self.scores = [evaluation.scores for evaluation in evaluations]
        self.targets = evaluations[0].targets

        self.model = ScoreCalibrationModel(len(evaluations))
        self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=0.01)

        self._prepare_training_data()

    def _prepare_training_data(self):
        self.target_llrs = None
        self.nontarget_llrs = None

        targets = torch.tensor(self.targets)
        
        for i, scores in enumerate(self.scores):
            scores = torch.tensor(scores, dtype=torch.float64)
            target_llrs_    = scores[targets == 1]
            nontarget_llrs_ = scores[targets == 0]

            if self.target_llrs is None:
                self.target_llrs = torch.empty(
                    (len(target_llrs_), len(self.scores))
                )
                self.nontarget_llrs = torch.empty(
                    (len(nontarget_llrs_), len(self.scores))
                )
            
            self.target_llrs[:, i] = target_llrs_
            self.nontarget_llrs[:, i] = nontarget_llrs_

    def train(self, epochs=50):
        losses = [cllr(self.target_llrs, self.nontarget_llrs)]

        for _ in range(epochs):
            def closure():
                self.optimizer.zero_grad()
                loss = cllr(
                    self.model(self.target_llrs),
                    self.model(self.nontarget_llrs)
                )
                loss.backward()
                return loss

            loss = self.optimizer.step(closure)
            
            if min(losses) - loss < 1e-4: break

            losses.append(loss.item())
        
    def predict(self, scores):
        return self.model(scores)