# Third-Party Notices

ThunderLlamaX is MIT-licensed. Portions of this project derive from, interoperate
with, or follow published specifications of the following MIT-licensed projects:

## tinygrad
- **Upstream:** https://github.com/tinygrad/tinygrad (MIT License)
- **Use:** `patches/tinygrad-fork.patch` modifies the tinygrad runtime (QMD/graph
  machinery, DriverKit dext integration, tripwires). The patch is a derivative work
  of tinygrad and is distributed under the terms of tinygrad's MIT license.
  The TinyGPU DriverKit dext (`extra/usbgpu` lineage) originates from the tinygrad
  project as well.

## llama.cpp / GGML
- **Upstream:** https://github.com/ggml-org/llama.cpp (MIT License)
- **Use:** GGUF container parsing, quantization formats (IQ4_XS, IQ3_S, IQ2_S,
  Q6_K, Q8_0, Q4_K), and dequantization reference implementations follow the
  published llama.cpp/GGML specifications; numpy dequant ports in this repository
  were validated bit-exactly against compiled llama.cpp. Format-level derivations
  are distributed under MIT in keeping with the upstream license.

## QuarkStar (Ninnix/q36)
- **Upstream:** https://github.com/Ninnix/q36
- **Use:** studied as a reference for MoE kernel organization (design reference
  only; no code copied).

## Model weights
- Model weight files (GGUF) are NOT part of this repository. Obtain them from
  their respective publishers (e.g., Qwen/Alibaba, Unsloth dynamic quants) and
  respect their licenses (Apache 2.0 for Qwen3-family weights).
