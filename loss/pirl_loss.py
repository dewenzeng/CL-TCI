import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .alias_multinomial import AliasMethod

# code adapted from https://github.com/HobbitLong/PyContrast/
class contrast_PIRL(nn.Module):
    """Memory bank for single modality"""
    def __init__(self, n_dim, n_data, K=65536, T=0.07, m=0.5, device='cuda:0'):
        super(contrast_PIRL, self).__init__()

        self.K = K
        self.T = T
        self.m = m
        self.device = device
        self.n_data = n_data
        # # create sampler
        self.multinomial = AliasMethod(torch.ones(n_data))
        self.multinomial.to(self.device)

        # create memory bank
        self.register_buffer('memory', torch.randn(n_data, n_dim))
        self.memory = F.normalize(self.memory)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def forward(self, x, y, x_jig, all_x=None, all_y=None):
        """
        Args:
          x: feat on current node
          y: index on current node
          x_jig: jigsaw feat on current node
          all_x: gather of feats across nodes; otherwise use x
          all_y: gather of index across nodes; otherwise use y
        """
        bsz = x.size(0)
        n_dim = x.size(1)

        # sample negative features
        idx = self.multinomial.draw(bsz * (self.K + 1)).view(bsz, -1)
        idx.select(1, 0).copy_(y.data)
        w = torch.index_select(self.memory, 0, idx.view(-1))
        w = w.view(bsz, self.K + 1, n_dim)

        # compute logits
        logits = self._compute_logit(x, w)
        logits_jig = self._compute_logit(x_jig, w)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).to(self.device)

        loss_1 = self.criterion(logits, labels)
        loss_2 = self.criterion(logits_jig, labels)
        loss = (1 - 0.5) * loss_1 + 0.5 * loss_2

        # update memory
        if (all_x is not None) and (all_y is not None):
            self._update_memory(self.memory, all_x, all_y)
        else:
            self._update_memory(self.memory, x, y)

        return loss

    def _update_memory(self, memory, x, y):
        """
        Args:
          memory: memory buffer
          x: features
          y: index of updating position
        """
        with torch.no_grad():
            x = x.detach()
            w_pos = torch.index_select(memory, 0, y.view(-1))
            w_pos.mul_(self.m)
            w_pos.add_(torch.mul(x, 1 - self.m))
            updated_weight = F.normalize(w_pos)
            memory.index_copy_(0, y, updated_weight)

    def _compute_logit(self, x, w):
        """
        Args:
          x: feat, shape [bsz, n_dim]
          w: softmax weight, shape [bsz, self.K + 1, n_dim]
        """
        x = x.unsqueeze(2)
        out = torch.bmm(w, x)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()
        return out

if __name__ == "__main__":
    contrastive_loss = contrast_PIRL(512, 25351, K=128, T=0.1, m=0.5, device='cuda:0').to('cuda:0')
    x = torch.randn([5,512]).to('cuda:0')
    y = torch.randint(0, 1000, (5,)).to('cuda:0')
    x_jig = torch.randn([5,512]).to('cuda:0')
    loss = contrastive_loss(x,y,x_jig)
    print(f'loss:{loss}')