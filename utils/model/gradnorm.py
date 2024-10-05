import torch


class GradNorm(torch.nn.Module):

    def __init__(self, model: torch.nn.Module, alpha: float = 1.0):
        super().__init__()
        self.init_task_loss = None
        # get layer of shared weights
        self.model = model
        # alpha hyper-param
        self.alpha = torch.tensor([alpha])

    def forward(self, task_loss, weights):
        self.alpha = self.alpha.to(weights.device)
        # get the gradient norms for each of the tasks
        norms = []
        for i in range(len(task_loss)):
            # get the gradient of this task loss with respect to the shared parameters
            if isinstance(self.model, torch.nn.DataParallel):
                parameters = self.model.module.get_last_layer().parameters()
            else:
                parameters = self.model.get_last_layer().parameters()
            gygw = torch.autograd.grad(task_loss[i], parameters, retain_graph=True)
            # compute the norm
            norms.append(torch.norm(torch.mul(weights[i], gygw[0])))
        norms = torch.stack(norms)
        # compute the inverse training rate r_i(t)
        loss_ratio = task_loss / self.init_task_loss
        inverse_train_rate = loss_ratio / torch.mean(loss_ratio)
        # compute the mean norm \tilde{G}_w(t)
        mean_norm = torch.mean(norms)
        # compute the GradNorm loss
        # this term has to remain constant
        constant_term = (mean_norm * (inverse_train_rate ** self.alpha)).clone().detach()
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
        # compute the gradient for the weights
        weights.grad = torch.autograd.grad(grad_norm_loss, weights)[0]

    def update_init_task_loss(self, init_task_loss):
        self.init_task_loss = init_task_loss

    def get_init_task_loss(self):
        return self.init_task_loss
