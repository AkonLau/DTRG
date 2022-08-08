import torch
import torch.nn as nn
import torch.nn.functional as F

class DTRG(nn.Module):
    def __init__(self, conf, feat_dim, use_gpu=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = conf.num_class
        self.start_dtrg = conf.start_dtrg
        self.ocl = conf.ocl
        self.weight_cent = conf.weight_cent
        self.graph = conf.graph
        self.distmethod = conf.distmethod
        self.tau = conf.tau
        self.eta = conf.eta
        self.use_gpu = use_gpu

        self.matrix = torch.randn((self.num_classes, feat_dim))
        self.grad = torch.zeros((self.num_classes, feat_dim))
        self.count = torch.zeros((self.num_classes, 1))

        if self.use_gpu:
            self.matrix = self.matrix.cuda()
            self.grad = self.grad.cuda()
            self.count = self.count.cuda()

        matrix_norm = self.matrix / torch.norm(self.matrix, keepdim=True, dim=-1)
        self.graph_weight_matrix = torch.mm(matrix_norm, matrix_norm.transpose(1, 0))

    def forward(self, xf, target, epoch):
        """
        Args:
            xf: feature matrix with shape (batch_size, feat_dim).
            target: ground truth labels with shape (batch_size).
            epoch: represent the current epoch
        """
        if self.training:
            with torch.no_grad():
                self.grad.index_add_(0, target, xf.detach())
                self.count.index_add_(0, target, torch.ones_like(target.view(-1, 1), dtype=torch.float32))

        if epoch >= self.start_dtrg:

            if self.ocl is True:
                centers = self.matrix[target]
                center_loss = torch.pow(xf - centers, 2).sum(-1).mean()
                center_loss *= self.weight_cent
            else:
                if self.use_gpu:
                    center_loss = torch.tensor(0).cuda()
                else:
                    center_loss = torch.tensor(0)

            if self.graph is True:

                xf_norm = xf / torch.norm(xf, keepdim=True, dim=-1)
                matrix_norm = self.matrix / torch.norm(self.matrix, keepdim=True, dim=-1)
                samples_similarity_matrix = torch.mm(xf_norm, matrix_norm.transpose(1, 0))

                similarity_matrix = self.graph_weight_matrix[target]
                if  self.distmethod == 'eu':
                    # Euclidean distance
                    P = torch.exp(similarity_matrix / self.tau)
                    Q = torch.exp(samples_similarity_matrix / self.tau)

                    euclidean_dist = torch.pow(P - Q, 2).sum(-1).mean()
                    similarity_loss = euclidean_dist * self.eta

                elif self.distmethod =='kl':
                    # KL divergence
                    P = F.softmax(similarity_matrix / self.tau, dim=-1)
                    Q = F.softmax(samples_similarity_matrix / self.tau, dim=-1)
                    KLDivLoss = F.kl_div(Q.log(), P, None, None, 'none').sum(-1).mean()
                    similarity_loss = KLDivLoss * self.eta
                else:
                    raise NameError("Warn: the similarity measure method {} is not existing!".format(self.distmethod))
            else:
                if self.use_gpu:
                    similarity_loss =  torch.tensor(0).cuda()
                else:
                    similarity_loss =  torch.tensor(0)
            return similarity_loss + center_loss

        else:
            if self.use_gpu:
                return torch.tensor(0).cuda()
            else:
                return torch.tensor(0)

    def update(self):
        # reset matrix
        index = torch.where(self.count > 0)[0]
        self.matrix[index] = (self.grad[index] / self.count[index]).detach()

        matrix_norm = self.matrix / torch.norm(self.matrix, keepdim=True, dim=-1)
        self.graph_weight_matrix = torch.mm(matrix_norm, matrix_norm.transpose(1, 0))

        # reset and update
        nn.init.constant_(self.grad, 0.)
        nn.init.constant_(self.count, 0.)

class DTRG_AUG(nn.Module):
    # DTRG with Inter-Class Relation Augmentation
    def __init__(self, conf, feat_dim, use_gpu=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = conf.num_class
        self.start_dtrg = conf.start_dtrg
        self.ocl = conf.ocl
        self.weight_cent = conf.weight_cent
        self.graph = conf.graph
        self.distmethod = conf.distmethod
        self.tau = conf.tau
        self.eta = conf.eta
        self.use_gpu = use_gpu
        self.review_mix_center = conf.review_mix_center

        self.matrix = torch.randn((self.num_classes, feat_dim))
        self.grad = torch.zeros((self.num_classes, feat_dim))
        self.count = torch.zeros((self.num_classes, 1))

        if use_gpu:
            self.matrix = self.matrix.cuda()
            self.grad = self.grad.cuda()
            self.count = self.count.cuda()

        matrix_norm = self.matrix / torch.norm(self.matrix, keepdim=True, dim=-1)
        self.graph_weight_matrix = torch.mm(matrix_norm, matrix_norm.transpose(1, 0))

    def forward(self, xf, target_a, target_b, lam_a, lam_b, epoch):
        """
        Args:
            xf: feature matrix with shape (batch_size, feat_dim).
            target_a: ground truth labels with shape (batch_size).
            target_b: shuffled ground truth labels with shape (batch_size).
            lam_a：lambda wight for target_a
            lam_b：lambda wight for target_b
            epoch: represent the current epoch
        """
        if self.training:
            self.grad.index_add_(0, target_a, lam_a.unsqueeze(1) * xf.detach())
            self.grad.index_add_(0, target_b, lam_b.unsqueeze(1) * xf.detach())
            self.count.index_add_(0, target_a, lam_a.unsqueeze(1))
            self.count.index_add_(0, target_b, lam_b.unsqueeze(1))

        if epoch >= self.start_dtrg:
            if self.ocl is True:
                centers_a = self.matrix[target_a]
                centers_b = self.matrix[target_b]
                if self.review_mix_center:
                    center_loss = torch.pow(xf -(lam_a.unsqueeze(-1) * centers_a + lam_b.unsqueeze(-1) * centers_b), 2).sum(-1).mean()
                else:
                    center_loss = (lam_a * (torch.pow(xf - centers_a, 2).sum(-1))).mean() \
                                  + (lam_b * (torch.pow(xf - centers_b, 2).sum(-1))).mean()
                center_loss *= self.weight_cent
            else:
                if self.use_gpu:
                    center_loss = torch.tensor(0).cuda()
                else:
                    center_loss = torch.tensor(0)

            if self.graph is True:
                similarity_matrix_a = self.graph_weight_matrix[target_a]
                similarity_matrix_b = self.graph_weight_matrix[target_b]

                xf_norm = xf / torch.norm(xf, keepdim=True, dim=-1)
                matrix_norm = self.matrix / torch.norm(self.matrix, keepdim=True, dim=-1)
                samples_similarity_matrix = torch.mm(xf_norm, matrix_norm.transpose(1, 0))

                if  self.distmethod == 'eu':
                    # Euclidean distance
                    P_a = torch.exp(similarity_matrix_a / self.tau)
                    P_b = torch.exp(similarity_matrix_b / self.tau)
                    Q = torch.exp(samples_similarity_matrix / self.tau)

                    dist_a = torch.pow(P_a - Q, 2).sum(-1)
                    dist_b = torch.pow(P_b - Q, 2).sum(-1)
                    mix_euclidean_dist = (lam_a * dist_a).mean() + (lam_b * dist_b).mean()

                    similarity_loss = mix_euclidean_dist * self.eta

                elif self.distmethod =='kl':
                    # KL divergence
                    P_a = F.softmax(similarity_matrix_a / self.tau, dim=-1)
                    P_b = F.softmax(similarity_matrix_b / self.tau, dim=-1)
                    Q = F.softmax(samples_similarity_matrix / self.tau, dim=-1)
                    KLDivLoss_a = F.kl_div(Q.log(), P_a, None, None, 'none').sum(-1)
                    KLDivLoss_b = F.kl_div(Q.log(), P_b, None, None, 'none').sum(-1)
                    mix_KLDivLoss = (lam_a * KLDivLoss_a).mean() + (lam_b * KLDivLoss_b).mean()
                    similarity_loss = mix_KLDivLoss * self.eta

                else:
                    raise NameError("Warn: the similarity measure method {} is not existing!".format(self.distmethod))

            else:
                if self.use_gpu:
                    similarity_loss =  torch.tensor(0).cuda()
                else:
                    similarity_loss =  torch.tensor(0)

            return similarity_loss + center_loss

        else:
            if self.use_gpu:
                return torch.tensor(0).cuda()
            else:
                return torch.tensor(0)

    def update(self):
        # reset matrix and update
        index = torch.where(self.count > 0)[0]
        self.matrix[index] = self.grad[index] / self.count[index]

        matrix_norm = self.matrix / torch.norm(self.matrix, keepdim=True, dim=-1)
        self.graph_weight_matrix = torch.mm(matrix_norm, matrix_norm.transpose(1, 0))

        # reset
        nn.init.constant_(self.grad, 0.)
        nn.init.constant_(self.count, 0.)