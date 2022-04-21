import argparse
import deepspeed
import json
import random
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models


# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark')
parser.add_argument('--model', type=str, default='resnet50',
                    help='Name of the model from torchvision')
parser.add_argument('--num-iters', type=int, default=50,
                    help='number of benchmark iterations')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# only to get the batch size from teh config file
with open('ds_config.json') as fp:
     ds_config = json.load(fp)

train_batch_size = ds_config['train_batch_size']

model = getattr(models, args.model)()

class SyntheticDataset(Dataset):
    def __getitem__(self, idx):
        data = torch.randn(3, 224, 224)
        target = random.randint(0, 999)
        return (data, target)

    def __len__(self):
        return train_batch_size * args.num_iters


train_set = SyntheticDataset()

parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=parameters,
    training_data=train_set
)

for epoch in range(1):
    for step, batch in enumerate(trainloader):
        imgs, labels = (batch[0], batch[1].to(model_engine.local_rank))
        if model_engine.fp16_enabled():
            imgs = imgs.half()

        imgs = imgs.to(model_engine.local_rank)

        outputs = model(imgs)
        loss = F.cross_entropy(outputs, labels)
        model_engine.backward(loss)
        model_engine.step()
