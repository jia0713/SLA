# SLA
This repository provides the implementation of [SLA](https://www.arxiv.org/pdf/2509.24006) (Sparse–Linear Attention), a trainable attention method that fuses sparse and linear attention to accelerate diffusion models.

SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention  
*Jintao Zhang, Haoxu Wang, Kai Jiang, Shuo Yang, Kaiwen Zheng, Haocheng Xi, Ziteng Wang, Hongzhou Zhu, Min Zhao, Ion Stoica, Joseph E. Gonzalez, Jianfei Chen, Jun Zhu*  
Paper: https://www.arxiv.org/pdf/2509.24006  

We provide two versions of SLA: you can use either the [SLA-1 branch](https://github.com/thu-ml/SLA/tree/SLA-1) or the [main branch](https://github.com/thu-ml/SLA).


![SLA Overview](./assets/overview_of_SLA.png)

### Motivation
![SLA Motivation](./assets/SLA_motivation.png)

### Effectiveness
![SLA Effectiveness](./assets/SLA_effectiveness.png)

### Efficiency
![SLA Efficiency](./assets/SLA_efficiency.png)


### Installation

```bash
git clone https://github.com/thu-ml/SLA.git
cd SLA
pip install -e .
```

### Usage

```python
import torch
from sparse_linear_attention import SparseLinearAttention

attn = SparseLinearAttention(
    head_dim=128,
    topk=0.2,                 # = 1 - sparsity
    feature_map="softmax",    # options: elu, relu, softmax
    BLKQ=64,
    BLKK=64,
).cuda()

B, H, L, D = 2, 4, 4096, 128
q = torch.randn((B, H, L, D), dtype=torch.bfloat16, device='cuda')
k = torch.randn((B, H, L, D), dtype=torch.bfloat16, device='cuda')
v = torch.randn((B, H, L, D), dtype=torch.bfloat16, device='cuda')

o = attn(q, k, v)
```

### MACA Sparse Attention Development

The `maca` branch includes a MACA MMA sparse attention forward path and a
benchmark that compares it with the Triton implementation. The local machine is
used for editing; build and test commands below should be run on a MACA GPU
machine.

Sync the local checkout to the MACA host:

```bash
rsync -az \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  --exclude build \
  --exclude dist \
  --exclude '*.egg-info' \
  /home/jianing/workspace/SLA/ \
  acl_ici@10.0.180.24:/home/acl_ici/workspace/SLA/
```

Prepare the remote environment:

```bash
ssh acl_ici@10.0.180.24
source /home/acl_ici/miniforge3/etc/profile.d/conda.sh
conda activate sla

cd /home/acl_ici/workspace/SLA

export MACA_PATH=/opt/maca
export PATH=/opt/maca/mxgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=/home/acl_ici/miniforge3/envs/sla/lib/python3.12/site-packages/torch/lib:/opt/maca/lib:$LD_LIBRARY_PATH
export MAX_JOBS=4
```

Build the extension:

```bash
python setup.py build_ext --inplace
```

Run correctness tests:

```bash
python -m pytest tests/test_cuda_sparse_attn.py -q
```

Run the full MACA MMA and sparse attention test set:

```bash
python -m pytest \
  tests/test_maca_mma_layout_probe.py \
  tests/test_maca_attention_tile_probe.py \
  tests/test_cuda_sparse_attn.py \
  -q
```

Run the Triton-vs-MACA performance benchmark:

```bash
python -m evaluate.bench_maca_sparse_attn \
  --batch 1 \
  --heads 2 \
  --seqlens 512 1024 2048 \
  --head-dim 64 \
  --block-m 64 \
  --topk-ratio 0.5 \
  --warmup 5 \
  --iters 10
```

For a larger multi-head case:

```bash
python -m evaluate.bench_maca_sparse_attn \
  --batch 2 \
  --heads 16 \
  --seqlens 512 1024 2048 \
  --head-dim 64 \
  --block-m 64 \
  --topk-ratio 0.25 \
  --warmup 5 \
  --iters 10
```

In the benchmark output, `triton_ms` is the Triton baseline, `maca_ms` is the
MACA kernel time, and `speedup > 1.0x` means the MACA kernel is faster than
Triton.

### SageSLA

We provide **SageSLA**, a very fast SLA (Sparse-Linear Attention) forward pass based on [SageAttention](https://github.com/thu-ml/SageAttention). It uses some code from [SpargeAttn](https://github.com/thu-ml/SpargeAttn). Please refer to the `SageSLA/` directory for the usage of SageSLA.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention},
  author={Zhang, Jintao and Wang, Haoxu and Jiang, Kai and Yang, Shuo and Zheng, Kaiwen and Xi, Haocheng and Wang, Ziteng and Zhu, Hongzhou and Zhao, Min and Stoica, Ion and others},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}

@article{zhang2026sla2,
  title={SLA2: Sparse-Linear Attention with Learnable Routing and QAT},
  author={Zhang, Jintao and Wang, Haoxu and Jiang, Kai and Zheng, Kaiwen and Jiang, Youhe and Stoica, Ion and Chen, Jianfei and Zhu, Jun and Gonzalez, Joseph E},
  journal={arXiv preprint arXiv:2602.12675},
  year={2026}
}

@inproceedings{zhang2025sageattention,
  title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}, 
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
