import torch
from torch import nn
from rec.base.torch_base_model import BaseModel


class MLP(BaseModel):
    def __init__(self, features, *args, **kwargs):
        super(MLP, self).__init__(features, *args, **kwargs)
        #
        input_dim = self.concated_dim
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        #
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input_map):
        concated_embed = self.get_concated_embedding(input_map)
        # logits = self.mlp(concated_embed)
        y_pred = self.mlp(concated_embed)
        if 'label' in input_map:
            label = input_map['label']
            # y_pred = torch.sigmoid(logits)
            loss = self.bce_loss(y_pred.squeeze(-1), label)
            # loss = self.cross_entropy(y_pred, label.unsqueeze(-1))
            # breakpoint()
            return y_pred, loss
        # y_pred = torch.sigmoid(logits)
        return y_pred
