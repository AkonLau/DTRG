"""
Reference:
    paper: Delving Deep into Label Smoothing.
"""
import torch
import torch.nn as nn

class OnlineLabelSmoothing(nn.Module):
    def __init__(self, num_classes=10, use_gpu=False):
        super().__init__()
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes))
        self.grad = torch.zeros((num_classes, num_classes))
        self.count = torch.zeros((num_classes, 1))
        self.ce_criterion = nn.CrossEntropyLoss().cuda()
        if use_gpu:
            self.matrix = self.matrix.cuda()
            self.grad = self.grad.cuda()
            self.count = self.count.cuda()

    def forward(self, x, target):
        if self.training:
            # accumulate correct predictions
            p = torch.softmax(x.detach(), dim=1)
            _, pred = torch.max(p, 1)
            correct_index = pred.eq(target)
            correct_p = p[correct_index]
            correct_label = target[correct_index]

            self.grad[correct_label] += correct_p
            self.grad.index_add_(0, correct_label, correct_p)
            self.count.index_add_(0, correct_label, torch.ones_like(correct_label.view(-1, 1), dtype=torch.float32))


        target = target.view(-1,)
        logprobs = torch.log_softmax(x, dim=-1)

        softlabel = self.matrix[target]
        ols_loss = (- softlabel * logprobs).sum(dim=-1)

        loss = 0.5 * self.ce_criterion(x, target) + 0.5 * ols_loss.mean()

        return loss

    def update(self):
        index = torch.where(self.count > 0)[0]
        self.grad[index] = self.grad[index] / self.count[index]
        # reset matrix and update
        nn.init.constant_(self.matrix, 0.)
        norm = self.grad.sum(dim=1).view(-1, 1)
        index = torch.where(norm > 0)[0]
        self.matrix[index] = self.grad[index] / norm[index]
        # reset
        nn.init.constant_(self.grad, 0.)
        nn.init.constant_(self.count, 0.)
