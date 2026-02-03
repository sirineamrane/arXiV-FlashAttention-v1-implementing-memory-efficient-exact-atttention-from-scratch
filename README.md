# arXiV-FlashAttention-v1-implementing-memory-efficient-exact-atttention-from-scratch

overview

this project reproduces flashattention v1 (arxiv: 2205.14135), a fast and memory-efficient exact attention algorithm for transformers. the goal is to understand the paper mathematically, implement a simplified version from scratch using python/triton, and benchmark against naive attention. this showcases graduate-level skills in ml theory, gpu optimization, and reproducible research


paper summary

title: flashattention: fast and memory-efficient exact attention with io-awareness
arxiv: 2205.14135

authors: tri dao, daniel y. fu, stefano ermon, atri rudra, christopher ré

flashattention addresses the bottleneck in transformer attention: the quadratic memory and runtime cost of computing softmax attention. the main contributions are:

io-aware attention: reduce memory bandwidth usage by tiling queries, keys, and values.

online softmax computation: compute softmax incrementally to avoid storing the full attention matrix.

backward recomputation: trade compute for memory in backprop to fit larger sequences on gpu.

this method achieves large speedups and memory savings compared to naive attention, making it the foundation of modern transformer optimization.


implementation

- data preprocessing / toy input: random sequences to test attention kernels.

- naive attention baseline: full softmax matrix computation.

- flashattention v1 simplified implementation:

--------online softmax

---------tiled computation

---------memory-efficient backward pass (simplified)

- metrics computation: speed, memory usage, correctness compared to baseline.

technologies used: python, pytorch, triton (optional for gpu kernels), matplotlib (for benchmarking plots).



benchmark & comparison

- naive attention: baseline runtime and memory.

- flashattention v1 simplified: faster runtime, lower memory footprint.

- optional comparison to flashattention-2: discussion of advanced gpu optimizations, occupancy improvements, and further speedups (no full implementation).

plots and tables are provided to clearly show performance gains.



discussion

- memory vs compute trade-offs in attention.

- io-aware tiling and online softmax reduce memory bottlenecks on modern gpus.

- limitations: toy implementation may not reach full industrial speedups; flashattention-2 introduces more optimizations for large-scale gpus.

- this project demonstrates understanding of transformer attention internals, gpu-aware algorithms, and reproducing a recent ml paper from scratch.
