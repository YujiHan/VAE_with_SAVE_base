import os
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import scanpy as sc
import torch.nn.functional as F
from collections import defaultdict
from torch.distributions import Normal, kl_divergence
from torch.optim import lr_scheduler

from model_VAE import VAE
from dataloader_VAE import get_h5ad_data, get_dataloader, normalize, inverse_normalize


def kl_div(mu, var):
    return (
        kl_divergence(
            Normal(mu, var.sqrt()), Normal(torch.zeros_like(mu), torch.ones_like(var))
        )
        .sum(dim=1)
        .mean()
    )


def train_vae(
    model,
    dataloader,
    num_epoch,
    kl_scale=0.5,
    device=torch.device("cuda:0"),
    lr=2e-4,
    seed=1234,
    is_lr_scheduler=True,
    weight_decay=5e-4,
):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, last_epoch=-1
    )

    tq = tqdm(range(num_epoch), ncols=80)
    for epoch in tq:
        model.train()
        epoch_loss = defaultdict(float)
        for i, (x, y) in enumerate(dataloader):
            x = x.float().to(device)

            z, mu, var = model.encoder(x)
            recon_x = model.decoder(z)

            # using bce loss estimating the error
            recon_loss = F.binary_cross_entropy(recon_x, x) * x.size(-1)
            kl_loss = kl_div(mu, var)
            loss = {"recon_loss": recon_loss, "kl_loss": kl_scale * kl_loss}

            optimizer.zero_grad()
            sum(loss.values()).backward()
            optimizer.step()

            for k, v in loss.items():
                epoch_loss[k] += loss[k].item()

        if is_lr_scheduler:
            scheduler.step()

        epoch_loss = {k: v / (i + 1) for k, v in epoch_loss.items()}
        epoch_info = ",".join(["{}={:.3f}".format(k, v) for k, v in epoch_loss.items()])
        tq.set_postfix_str(epoch_info)

    # for some config record
    return epoch_loss


def main():
    num_epoch = 20
    batch_size = 64

    data_list = get_h5ad_data()
    norm_data_list, scalers = normalize(data_list)
    dataloader = get_dataloader(norm_data_list, batch_size=batch_size)

    device = torch.device("cuda:1")
    net = VAE().to(device)

    train_vae(net, dataloader, num_epoch, device=device)


if __name__ == '__main__':
    main()
