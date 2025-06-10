#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import math
import time
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity_target=0.9, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_target = sparsity_target
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.reset_parameters()
        mask = (torch.rand(self.weight.shape) > self.sparsity_target).float()
        self.register_buffer('mask', mask)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

    def apply_l1_regularization(self, l1_lambda=1e-5):
        return l1_lambda * torch.norm(self.weight * self.mask, p=1)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim,
                 final_activation='linear', sparsity_ae=0.0):
        super().__init__()
        Layer = SparseLinear if sparsity_ae > 0 else nn.Linear
        enc_layers = []
        cur = input_dim
        for h in hidden_dims:
            enc_layers += [Layer(cur, h), nn.ReLU()]
            cur = h
        enc_layers.append(Layer(cur, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        cur = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [Layer(cur, h), nn.ReLU()]
            cur = h
        dec_layers.append(Layer(cur, input_dim))
        if final_activation == 'sigmoid':
            dec_layers.append(nn.Sigmoid())
        elif final_activation == 'tanh':
            dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def l1_regularization(self, l1_lambda=1e-5):
        loss = 0.0
        for m in self.modules():
            if isinstance(m, SparseLinear):
                loss += m.apply_l1_regularization(l1_lambda)
        return loss

def pretrain_autoencoder(model_ae, data_tensor, epochs, batch_size, lr, device, l1_lambda=0.0):
    model_ae.to(device).train()
    opt = torch.optim.Adam(model_ae.parameters(), lr=lr)
    ds = TensorDataset(data_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for (x,) in dl:
            x = x.to(device)
            opt.zero_grad()
            x_recon,_ = model_ae(x)
            loss = F.mse_loss(x_recon, x)
            if l1_lambda>0:
                loss += model_ae.l1_regularization(l1_lambda)
            loss.backward()
            opt.step()
    return model_ae


def load_all_MA(folder):
    runs_M, runs_A = [], []
    genes, samples = None, None
    files = os.listdir(folder)
    idxs = sorted({fn[len("proc_"):-len("_S.csv")]
                   for fn in files if fn.startswith("proc_") and fn.endswith("_S.csv")
                   and f"proc_{fn[len('proc_'):-len('_S.csv')]}_A.csv" in files},
                  key=int)
    for i in idxs:
        dfS = pd.read_csv(os.path.join(folder,f"proc_{i}_S.csv"), index_col=0)
        dfA = pd.read_csv(os.path.join(folder,f"proc_{i}_A.csv"), index_col=0)
        if genes is None:   genes   = dfS.index.to_list()
        if samples is None: samples = dfA.index.to_list()
        runs_M.append(dfS.values.astype(np.float32))
        runs_A.append(dfA.values.astype(np.float32))
    return runs_M, runs_A, genes, samples


def flatten_M_vectors(runs_M):
    m_list, ids = [], []
    for r,M in enumerate(runs_M):
        for c in range(M.shape[1]):
            m_list.append(M[:,c])
            ids.append((r,c))
    return np.stack(m_list,axis=0), ids


def build_consensus(runs_M, runs_A, ids, labels, genes, samples, active_clusters):

    clusters = [int(c) for c in active_clusters]
    G, N = len(genes), len(samples)
    K = len(clusters)
    Mc = np.zeros((G, K), dtype=np.float32)
    Ac = np.zeros((N, K), dtype=np.float32)
    for j, cl in enumerate(clusters):
        sel = np.where(labels == cl)[0]
        Ms = np.stack([runs_M[ids[i][0]][:, ids[i][1]] for i in sel], axis=1)
        As = np.stack([runs_A[ids[i][0]][:, ids[i][1]] for i in sel], axis=1)
        ref = Ms[:, 0]
        for k in range(Ms.shape[1]):
            if np.corrcoef(ref, Ms[:, k])[0,1] < 0:
                Ms[:, k] *= -1
                As[:, k] *= -1
        Mc[:, j] = Ms.mean(axis=1)
        Ac[:, j] = As.mean(axis=1)
    dfM = pd.DataFrame(Mc, index=genes, columns=[f"cl_{c}" for c in clusters])
    dfA = pd.DataFrame(Ac, index=samples, columns=[f"cl_{c}" for c in clusters])
    return dfM, dfA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',         required=True,
                        help="proc_{i}_S.csv & proc_{i}_A.csv Path")
    parser.add_argument('--max_k',       type=int, default=200,
                        help="DP-GMM highest threshold")
    parser.add_argument('--dp_prior',    type=float, default=1e-2,
                        help="DP-GMM concentration prior")
    parser.add_argument('--ae_epochs',   type=int, default=30)
    parser.add_argument('--batch_size',  type=int, default=128)
    parser.add_argument('--lr_nn',       type=float, default=1e-3)
    parser.add_argument('--threshold', type=float, default=0.001,
                        help="keep weight > threshold as ")
    parser.add_argument('--output_pref', type=str, default='robust')
    parser.add_argument('--run_frac',   type=float, default=0.5,
                        help="Activate ratio threshold")
    args = parser.parse_args()

    runs_M, runs_A, genes, samples = load_all_MA(args.dir)
    M_flat, ids = flatten_M_vectors(runs_M)

    scaler = StandardScaler().fit(M_flat)
    M_norm = scaler.transform(M_flat)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_tensor = torch.tensor(M_norm, dtype=torch.float32).to(device)

    ae_model = AutoEncoder(input_dim=M_norm.shape[1],
                           hidden_dims=[512,256],
                           latent_dim=50,
                           final_activation='linear',
                           sparsity_ae=0.0).to(device)

    ae_model = pretrain_autoencoder(ae_model, data_tensor,
                                    epochs=args.ae_epochs,
                                    batch_size=args.batch_size,
                                    lr=args.lr_nn,
                                    device=device)

    ae_model.eval()
    zs = []
    with torch.no_grad():
        dl = DataLoader(TensorDataset(data_tensor), batch_size=args.batch_size)
        for (x_batch,) in dl:
            _, z_batch = ae_model(x_batch)
            zs.append(z_batch.cpu().numpy())
    Z = np.concatenate(zs, axis=0)  
    dp = BayesianGaussianMixture(
        n_components=args.max_k,
        covariance_type='diag',
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1e-15,
        max_iter=1000,
        random_state=0
    )
    dp.fit(Z)
    labels = dp.predict(Z)
    print(np.unique(labels))
    weights = dp.weights_          # shape (max_k,)

    active_by_weight = np.where(weights > 0)[0]

    n_runs = len(runs_M)
    robust = []
    for cl in active_by_weight:
        sel = np.where(labels == cl)[0]
        runs_in_cl = { ids[i][0] for i in sel }
        if len(runs_in_cl) / n_runs >= args.run_frac:
            robust.append(cl)

    robust = np.array(sorted(robust))
    print(f"Keep {len(robust)} clusters：", robust.tolist())

    # 6) consensus 平均
    dfM_cons, dfA_cons = build_consensus(
        runs_M, runs_A, ids, labels, genes, samples, robust
    )

    # 7) 保存
    outM = os.path.join(args.dir, f"{args.output_pref}_M_consensus.csv")
    outA = os.path.join(args.dir, f"{args.output_pref}_A_consensus.csv")
    dfM_cons.to_csv(outM)
    dfA_cons.to_csv(outA)
    print("Saved: ", outM, outA)

if __name__ == '__main__':
    main()