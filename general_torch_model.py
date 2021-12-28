import torch
import numpy as np
import torch.nn as nn
from autoattack import square

class GeneralTorchModel(nn.Module):
    def __init__(self, model, n_class=10, im_mean=None, im_std=None):
        super(GeneralTorchModel, self).__init__()
        self.model = model
        self.model.eval()
        self.num_queries = 0
        self.im_mean = im_mean
        self.im_std = im_std
        self.n_class = n_class


    def forward(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        logits = self.model(image)
        return logits

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            processed = torch.from_numpy(image).type(torch.FloatTensor)
        else:
            processed = image

        if self.im_mean is not None and self.im_std is not None:
            im_mean = torch.tensor(self.im_mean).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            im_std = torch.tensor(self.im_std).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            processed = (processed - im_mean) / im_std
        return processed

    def predict_prob(self, image):

        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        logits = self.model(image)
        self.num_queries += image.size(0)
        return logits

    def predict_label(self, image):
        logits = self.predict_prob(image)
        _, predict = torch.max(logits, 1)
        return predict
def normalize(x,ndims):
    t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    print(t)
    a = t.view(-1, *([1] * ndims))
    print(a.shape)
    return x / (t.view(-1, *([1] * ndims)) + 1e-12)
if __name__ == "__main__":
    batch_view = lambda tensor: tensor.view(-1, *[1] * 3)
    a = torch.ones(1,3,2,2)
    print(a)
    delta = torch.ones_like(a)
    delta_l2_norm = delta.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
    print(delta_l2_norm)
    delta = delta/(delta_l2_norm.view(-1,1,1,1) + 1e-12)
    print(delta)
    print(delta.flatten(1).norm(p=2,dim=1))

    epsilon1 = 0.03
    epsilon2 = torch.tensor([0.5])

    # stepsize = epsilon2/delta
    # print(stepsize)
    # new_delta = stepsize * delta
    l2_norms = delta.flatten(1).norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
    print(l2_norms)
    new_delta = (delta.flatten(1) * (epsilon2.unsqueeze(1)).clamp_max(1)).view_as(delta)
    print(new_delta.flatten(1).norm(p=2,dim=1))
    # new_delta = 0.5 * delta
    # print(new_delta.flatten(1).norm(p=1,dim=1))
    # a = torch.full((1,3,2,2),0.2887)
    # print(a.flatten(1).norm(p=2,dim=1))
