# One-Page Comparison Template: DoSSR + Text-Region Consistency Loss vs. CCSR, PNFS, and TADiSR

**Owner:** Member 4 (`feat/report-figures`)  
**Status:** Structured template (`TODO: confirm` all paper-specific numbers and citations from the primary papers before finalizing).

> [!IMPORTANT]
> **Rule from `context.md`:** Do **not** guess or invent facts about **CCSR**, **PNFS**, or **TADiSR**. Every cell marked `TODO: confirm` must be filled directly from the published paper or official repository of that method.

---

## 1. High-Level Comparison Table

| Method | Primary Goal | Base Backbone | How It Handles Seed / Sampling Consistency or Text Regions | Training vs. Inference Cost | Key Difference vs. Our Approach |
|---|---|---|---|---|---|
| **DoSSR (Baseline)**<br>*(Verified in `external/DOSSR`)* | Fast diffusion real-world SR (`5` steps) | Stable Diffusion 2.1 + ControlNet + RealESRNet preprocessor | Domain-shift schedule (`trunc_12`) + 3rd-order `shift_sample_3_order`. **No text-region loss and no cross-seed consistency loss.** Restores `"eduroam"` as `"odaroam"` on `Canon_001_LR4`. | Inference: `5` steps (`~1.75–2.01 s/img` at `128->512`, FP16). | Our baseline model. |
| **CCSR**<br>*(`TODO: confirm` full title & venue)* | Content-consistent real-world super-resolution (`TODO: confirm` exact paper formulation) | Diffusion SR backbone (`TODO: confirm` exact SD version & control module) | `TODO: confirm` how CCSR enforces content/structure consistency across diffusion steps or samples. | `TODO: confirm` training stages and inference step count. | `TODO: confirm` whether CCSR uses a general whole-image consistency criterion vs. our localized text-box cross-noise $\hat{x}_0$ consistency loss. |
| **PNFS**<br>*(`TODO: confirm` full title & venue)* | `TODO: confirm` (check whether PNFS operates at test time in noise/feature space or during training) | `TODO: confirm` backbone | `TODO: confirm` exact mechanism for noise/seed/text handling. | `TODO: confirm` test-time overhead vs. training overhead. | Our method adds **zero extra inference steps** (keeps `5` steps) by learning LoRA weights with a 2-noise text-region loss during training. `TODO: confirm` contrast with PNFS. |
| **TADiSR**<br>*(`TODO: confirm` full title & venue)* | Text-aware diffusion super-resolution (`TODO: confirm` exact paper formulation) | `TODO: confirm` backbone and text detection/recognition modules | `TODO: confirm` whether TADiSR requires an external OCR/text-encoder module or extra conditioning at inference time. | `TODO: confirm` extra parameters and inference latency. | Our Phase 1 approach requires **no OCR or bounding boxes at test time**: the HR text-box mask is used **only in the training loss** ($\mathcal{L}_{\text{text-cons}}$), while LoRA `(B vs C)` isolates the effect of cross-noise consistency. |
| **Ours (DoSSR + Text-Region Consistency LoRA `C`)** | Seed-consistent text restoration in 5-step DoSSR without changing inference | Frozen DoSSR (`ControlLDM`) + lightweight LoRA (`r=4` or `8`) | Draws **2 independent noise samples** $(\epsilon_1, \epsilon_2)$ at the same timestep $t$ during LoRA training and penalizes $\|M_{\text{lat}} \odot (\hat{x}_0^{(1)} - \hat{x}_0^{(2)})\|_1$ inside HR text boxes. | Inference: **unchanged** (`5` steps, no test-time OCR/mask needed). Training: 2 forward passes/step on LoRA adapters only. | Directly targets seed-induced character hallucinations (`eduroam` $\rightarrow$ `odaroam`) via a controlled `(A, B, C)` ablation. |

---

## 2. Short Narrative Paragraphs for the Next Report (Draft Structure for Member 4)

### 2.1 Problem Statement: Seed Instability in Diffusion Scene-Text SR
- Diffusion SR models inject random Gaussian noise (`torch.randn_like` in `shift_sampler.py:346, 363`).
- On `RealSRVal_crop128/test_LR` (`128x128 -> 512x512`), changing the seed from `42` to `43` shifts output pixels by `mean_abs_diff = 8.82` (`max = 255.0`), and on `Canon_001_LR4` the sign `"eduroam"` is hallucinated as `"odaroam"`.

### 2.2 How Prior Methods Address Consistency and Text (`CCSR`, `PNFS`, `TADiSR`)
- **CCSR:** `TODO: confirm 2–3 short sentences summarizing CCSR from its paper.`
- **PNFS:** `TODO: confirm 2–3 short sentences summarizing PNFS from its paper.`
- **TADiSR:** `TODO: confirm 2–3 short sentences summarizing TADiSR from its paper.`

### 2.3 Why Our `(A, B, C)` LoRA Ablation Is Clean and Testable
- Unlike methods that add test-time OCR modules or extra sampling steps, our loss uses HR text boxes **only during LoRA training**.
- Because LoRA `(A)`, `(B)`, and `(C)` all draw **2 noise samples per step** with the same data, prompt, and budget, comparing **(B) higher text-region weight** against **(C) higher text-region weight + cross-noise $\hat{x}_0$ consistency loss** directly answers whether penalizing cross-noise disagreement improves character accuracy and seed stability.
