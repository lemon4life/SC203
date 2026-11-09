# Related Methods: Stability and Text Fidelity in Diffusion SR

| Method | Mechanism | Relation to our proposed text-region consistency loss |
|---|---|---|
| **CCSR** | Uses diffusion for structure recovery, then adversarially fine-tunes the VAE decoder for deterministic detail enhancement. Its non-uniform timestep strategy supports flexible inference step counts, including one or two. | Targets general content stability and sampling efficiency; our loss directly penalizes cross-noise disagreement inside annotated text regions. |
| **PNFS** | Trains a discrepancy predictor from LR images, initial noise, and HR-derived pseudo-labels. At inference, it probabilistically fuses elements of candidate initial noises before diffusion sampling. | Improves general SR by changing inference noise; our proposed method trains text-region consistency into LoRA weights and keeps the baseline inference path. |
| **TADiSR** | Fine-tunes latent-diffusion cross-attention with LoRA and replaces the standard decoder with interacting image-SR and text-segmentation decoders. It jointly outputs an SR image and text mask; training uses synthetic FTSR and filtered Real-CE pairs. | The closest task match: it explicitly models text, but changes attention/decoder architecture. Our experiment isolates a text-masked cross-noise loss while retaining the DoSSR architecture. |

CCSR and PNFS address seed sensitivity at the whole-image level: CCSR separates stochastic structure recovery from deterministic detail synthesis, while PNFS predicts and fuses favorable elements from multiple initial noises. TADiSR directly targets text fidelity through text-aware attention and joint segmentation decoding. The proposed DoSSR follow-up tests a narrower training change: whether consistency between independent noise predictions inside HR text masks improves text-region stability without adding test-time OCR or changing the five-step inference procedure.

The accompanying seed-42 figure is qualitative only: both generated crops visibly differ from the HR sign text. Whole-image PSNR, SSIM, and LPIPS averages do not establish character correctness; that requires the separate text-region evaluation.

## Primary papers

1. Sun et al., [Improving the Stability and Efficiency of Diffusion Models for Content Consistent Super-Resolution](https://arxiv.org/abs/2401.00877), arXiv:2401.00877, 2024.
2. Jiang et al., [Restabilizing Diffusion Models with Predictive Noise Fusion Strategy for Image Super-Resolution](https://doi.org/10.1609/aaai.v39i4.32421), AAAI 2025, 39(4):4021–4029.
3. Hu et al., [Text-Aware Real-World Image Super-Resolution via Diffusion Model with Joint Segmentation Decoders](https://proceedings.neurips.cc/paper_files/paper/2025/hash/58ce6a4b9c16d11975f11e4a23871041-Abstract-Conference.html), NeurIPS 2025.
