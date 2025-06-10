# DeepCluster_ICA

**DeepCluster-ICA** aggregates multiple FastICA runs into a **robust set of iModulons**  
(consensus gene-regulatory modules). The core pipeline is

> ICA → column-wise z-score → AutoEncoder (AE) dimension reduction → DP-GMM clustering  
>  → cross-run consistency filter → polarity correction + averaging → consensus *M* / *A* matrices

---
its a faster improvments based on iModulonMiner on getting consensus modules, it's more automatic without needs to compute and save distances
 between proc_{i}_s ,which save more compute resources

 
> **Input expectation**  
> For every ICA restart you need a *paired* `proc_<id>_S.csv` and `proc_<id>_A.csv`  
> (identical `<id>`). The folder name `ica_runs/` is arbitrary and can be changed  
> via `--dir` when running `dp_gmm.py`.

---

## Installation

```bash
conda create -n deepcica python=3.9
conda activate deepcica
pip install -r requirements.txt
```

If CUDA is available, the PyTorch wheel will automatically use the GPU; otherwise
the pipeline runs on CPU.

## One-liner usage
python dp_gmm.py \
  --dir ica_runs \
  --max_k 200 \
  --dp_prior 1e-2 \
  --threshold 1e-8 \
  --run_frac 0.5 \
  --output_pref robust

## Outputs
ica_runs/robust_M_consensus.csv   # genes × K robust clusters
ica_runs/robust_A_consensus.csv   # samples × K

python plot_dims.py --runs_dir ica_runs      # visualise component counts vs. dimension
python metrics_plot.py                       # quality metrics (edit paths inside script)

# Key command-line options (dp_gmm.py)
| flag            | default | purpose                                                   |
| --------------- | ------- | --------------------------------------------------------- |
| `--max_k`       | 200     | upper bound for DP-GMM components                         |
| `--dp_prior`    | 1e-2    | Dirichlet concentration; smaller → fewer clusters         |
| `--threshold`   | 1e-8    | keep clusters with mixture weight `w_k > threshold`       |
| `--run_frac`    | 0.5     | keep clusters that appear in ≥ `run_frac` of ICA restarts |
| `--ae_epochs`   | 30      | AutoEncoder pre-training epochs                           |
| `--batch_size`  | 128     | batch size for AE and inference                           |
| `--lr_nn`       | 1e-3    | learning rate for the AutoEncoder                         |
| `--output_pref` | robust  | prefix for consensus file names                           |

## Citation
If you use DeepCluster-ICA in your research, please cite:
@misc{DeepClusterICA2025,
  title        = {DeepCluster-ICA},
  author       = {Your Name},
  howpublished = {https://github.com/yourhandle/deepica},
  year         = {2025},
  note         = {v1.0.0}
}


