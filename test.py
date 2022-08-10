import torch

from torchmetrics import F1Score
target = torch.tensor([0, 1, 2, 0, 1, 2])
preds = torch.tensor([0, 2, 1, 0, 0, 1])
f1 = F1Score(num_classes=3,average='micro')
print(type(f1(preds, target)))
