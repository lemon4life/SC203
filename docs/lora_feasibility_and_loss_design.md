# Tasks C.9 & C.10: DoSSR Training Audit, Text-Region Loss Design, and LoRA Trial Estimates

**Status:** Read-only codebase audit complete (`VERIFIED` in `external/DOSSR`). All LoRA parameter, batch size, and VRAM numbers in Section 6 are **`ESTIMATE — NOT TESTED`**.

---

## 1. Executive Summary (Task C.9)

1. **Yes, `external/DOSSR` has a complete training entry point**, not only `shared_step` and a dataset class:
   - Entry script: [`external/DOSSR/scripts/train.py:1-30`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/scripts/train.py#L1-L30)
   - Main training config: [`external/DOSSR/configs/train_cldm.yaml:1-47`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/train_cldm.yaml#L1-L47)
   - Dataset config: [`external/DOSSR/configs/dataset/general_deg_realesrgan_train.yaml:1-62`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/dataset/general_deg_realesrgan_train.yaml#L1-L62)
   - Model config: [`external/DOSSR/configs/model/cldm_v21.yaml:1-95`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/model/cldm_v21.yaml#L1-L95)
2. **Prompt mismatch between upstream training and upstream inference:**
   - **Training** uses the **empty prompt `""`** for every sample ([`external/DOSSR/dataset/realesrgan.py:179`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/realesrgan.py#L179)).
   - **Inference** uses the **default prompt `"clean, high-resolution, 8k"`** ([`external/DOSSR/model/shift_sampler.py:200, 257, 322`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/shift_sampler.py#L322)).
3. **Clean-latent prediction ($\hat{x}_0$) is already computed inside `shared_step`:**
   - [`ControlLDM.shared_step`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/cldm.py#L346-L356) calls [`LatentDiffusion.predict_x0_out_mix(x, preprocess_x, c, t, noise=...)`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L909-L917) and computes `loss_preprocess_diff = (x - pred_x0).abs().mean()`.
   - Because `predict_x0_out_mix` accepts an explicit `noise` tensor, drawing two independent noise samples $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, I)$ at the same timestep $t$ is a direct 2-call wrapper.
4. **Two critical code caveats we must handle in our own launcher/wrapper (without editing `external/DOSSR`):**
   - **Caveat 1 (Training pool shuffles images without masks):** [`RealESRGANBatchTransform._dequeue_and_enqueue`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/batch_transform.py#L98-L134) keeps a 256-image GPU queue (`queue_lr`, `queue_gt`) and shuffles them with `torch.randperm` ([`line 117`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/batch_transform.py#L117)). It does not queue extra batch keys like a text mask. We must bypass `_dequeue_and_enqueue` (or use our own paired dataset/transform) so each HR text mask stays aligned with its image.
   - **Caveat 2 (`torch.no_grad()` inside `ControlledUnetModel`):** [`ControlledUnetModel.forward`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/cldm.py#L27-L35) wraps `self.input_blocks` and `self.middle_block` inside `with torch.no_grad():`. If we put LoRA adapters only in `self.control_model` (ControlNet) and/or `diffusion_model.output_blocks`, gradients flow normally. If we also put LoRA in `diffusion_model.input_blocks` or `middle_block`, our wrapper must override `ControlledUnetModel.forward` without `torch.no_grad()`.

---

## 2. Full Training Entry Point and Pipeline in `external/DOSSR`

### 2.1 Training Script and Configs
- **Entry script ([`external/DOSSR/scripts/train.py:10-30`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/scripts/train.py#L10-L30)):**
  1. Parses `--config` (`configs/train_cldm.yaml`).
  2. Calls `pl.seed_everything(config.lightning.seed, workers=True)` (`seed: 231` in [`train_cldm.yaml:2`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/train_cldm.yaml#L2)).
  3. Builds `data_module = instantiate_from_config(config.data)` $\rightarrow$ [`dataset.data_module.DataModule`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/data_module.py#L12-L70).
  4. Builds `model = instantiate_from_config(OmegaConf.load(config.model.config))` $\rightarrow$ [`model.cldm.ControlLDM`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/cldm.py#L244-L450).
  5. Loads `config.model.resume` via `load_state_dict(..., strict=True)`.
  6. Creates `pl.Trainer(**config.lightning, callbacks=callbacks, logger=logger)` and calls `trainer.fit(model, datamodule=data_module)` ([`line 26`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/scripts/train.py#L26)).

### 2.2 Dataset and GPU Degradation Pipeline
- **CPU dataset ([`external/DOSSR/dataset/realesrgan.py:86-180`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/realesrgan.py#L86-L180)):**
  - Loads HR images from a file list (`meta_info`), center-crops or random-crops to `out_size: 512` ([`general_deg_realesrgan_train.yaml:13`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/dataset/general_deg_realesrgan_train.yaml#L13)), generates blur kernels (`kernel1`, `kernel2`, `sinc_kernel`), and returns:
    ```python
    return_d = {
        "hq": img_hq,
        "kernel1": kernel,
        "kernel2": kernel2,
        "sinc_kernel": sinc_kernel,
        "txt": "",
    }
    ```
- **GPU batch transform ([`external/DOSSR/dataset/data_module.py:54-59`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/data_module.py#L54-L59) & [`dataset/batch_transform.py:136-270`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/batch_transform.py#L136-L270)):**
  - `DataModule.on_after_batch_transfer` runs `RealESRGANBatchTransform` on the GPU.
  - It synthesizes second-order Real-ESRGAN degradations (`128x128` LR), calls `_dequeue_and_enqueue()`, bicubic-upsamples the degraded LR back to `512x512` (`hint` in `[0, 1]`, NHWC), and formats the clean HR target (`jpg` in `[-1, 1]`, NHWC):
    ```python
    return {"jpg": gt, "hint": lq, "txt": batch["txt"]}
    ```
  - **Why we must not use `RealESRGANBatchTransform._dequeue_and_enqueue` for masked training:**
    - `_dequeue_and_enqueue` ([`batch_transform.py:98-134`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/batch_transform.py#L98-L134)) mixes the current batch with an internal pool (`queue_size: 256`) using `idx = torch.randperm(self.queue_size)`. It does **not** permute `batch["txt"]` or any custom `batch["text_mask"]`.
    - For our LoRA training with text boxes, our training dataset/transform in `src/` will load paired `(LR, HR, text_mask)` (or synthesize LR from HR without `_dequeue_and_enqueue`) so the text mask always matches the exact HR image.

---

## 3. Which Prompt Does DoSSR Training Use, and What Does That Mean for LoRA `(A, B, C)`?

1. **What upstream training uses:**
   - [`external/DOSSR/dataset/realesrgan.py:179`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/realesrgan.py#L179) sets `"txt": ""` (empty string) for every training example.
   - [`ControlLDM.get_input`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/cldm.py#L363-L366) encodes `batch["txt"]` with `self.get_learned_conditioning(xc)` (`FrozenOpenCLIPEmbedder`, [`configs/model/cldm_v21.yaml:84-88`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/model/cldm_v21.yaml#L84-L88)) and stores it in `c_crossattn`.
2. **What upstream inference uses:**
   - [`external/DOSSR/model/shift_sampler.py:200, 257, 322`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/shift_sampler.py#L322) defaults to `positive_prompt: str="clean, high-resolution, 8k"`.
   - [`external/DOSSR/scripts/inference.py:75-78`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/scripts/inference.py#L75-L78) calls `sampler.shift_sample_3_order(..., cfg_scale=1.0)` without passing a prompt, so it always feeds `"clean, high-resolution, 8k"` at test time.
3. **Prompt Proposal for our 3 LoRA versions `(A, B, C)` (`TODO: decision for user and mentor`):**
   - First, our 10-seed baseline study (Group A) compares `prompt_mode=default` (`"clean, high-resolution, 8k"`) vs `prompt_mode=empty` (`""`) on the frozen checkpoint.
   - **Proposal (`TODO: confirm with user and mentor`):** **Train LoRA versions `A`, `B`, and `C` with the empty prompt `""` (as DoSSR training does in [`dataset/realesrgan.py:179`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/realesrgan.py#L179)) and evaluate them with the same empty-prompt mode.**
   - Do **not** switch the LoRA prompt based on whichever prompt wins Group A without explicit confirmation from the user and mentor. All three versions (`A`, `B`, `C`) must use the exact same prompt mode and log the exact prompt string in `lora_run_manifest.json`.

---

## 4. Exact Loss and Clean-Latent ($\hat{x}_0$) Equations in `external/DOSSR`

### 4.1 `training_step` and `shared_step` ([`external/DOSSR/model/cldm.py:346-374`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/cldm.py#L346-L374))
```python
def shared_step(self, batch, **kwargs):
    x, c = self.get_input(batch, self.first_stage_key)
    preprocess_x = c['c_preprocess'][0]
    t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
    pred_x0 = self.predict_x0_out_mix(x, preprocess_x, c, t, **kwargs)
    loss_preprocess_diff = (x - pred_x0).abs().mean()
    loss = loss_preprocess_diff
    return loss, {'train/loss_preprocess_diff': loss_preprocess_diff}
```
Where:
- `x` ($x_0 \in \mathbb{R}^{B \times 4 \times 64 \times 64}$): Clean HR latent from `self.encode_first_stage(jpg)` scaled by `scale_factor = 0.18215` ([`ddpm.py:780-785`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L780-L785)).
- `preprocess_x` ($\hat{x}_{\text{LR}} \in \mathbb{R}^{B \times 4 \times 64 \times 64}$): Preprocessed LR latent from [`ControlLDM.preprocess`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/cldm.py#L338-L343):
  1. Bicubic downsample `hint` by $1/4$ to `128x128`.
  2. Run frozen `RealESRNet` (`self.preprocess_model`) to `512x512` and normalize to `[-1, 1]`.
  3. Encode with `self.cond_encoder(c).mode() * 0.18215`.
- `c`: Dictionary with:
  - `c_crossattn`: `[FrozenOpenCLIPEmbedder(txt)]` ($\in \mathbb{R}^{B \times 77 \times 1024}$)
  - `c_concat`: `[hint]` ($\in \mathbb{R}^{B \times 3 \times 512 \times 512}$)
  - `c_preprocess`: `[preprocess_x]` ($\in \mathbb{R}^{B \times 4 \times 64 \times 64}$)

### 4.2 How `pred_x0` Is Defined in `eps`-Prediction Mode with the Domain-Shift Schedule ([`external/DOSSR/ldm/models/diffusion/ddpm.py:909-917`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L909-L917))
- `DDPM.__init__` sets `self.parameterization = "eps"` ([`ddpm.py:75, 87`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L75)).
- `DDPM.register_schedule` ([`ddpm.py:162-164`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L162-L164)) registers the domain-shift schedule `self.lambdas = make_eta_schedule("trunc_12", timesteps)` ([`ldm/modules/diffusionmodules/util.py:29-48`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/modules/diffusionmodules/util.py#L29-L48)), where $\lambda_t \in [0, 0.999]$ increases linearly from $0$ at $t=0$ to $0.999$ at $t = 500$ and stays $0.999$ for $t \in [500, 999]$.
- Given HR latent $x_0$, preprocessed LR latent $\hat{x}_{\text{LR}}$, conditioning $c$, timestep $t$, and Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$, [`predict_x0_out_mix`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L909-L917) computes:
  1. **Domain-shifted clean latent interpolation ([`ddpm.py:913`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L913)):**
     $$x_{\text{rect}}(t) = \lambda_t \, \hat{x}_{\text{LR}} + (1 - \lambda_t) \, x_0$$
  2. **Forward noisy latent ([`ddpm.py:914`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L914) & [`q_sample` at `ddpm.py:365-368`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L365-L368)):**
     $$x_t(\epsilon) = \sqrt{\bar{\alpha}_t} \, x_{\text{rect}}(t) + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$$
  3. **UNet + ControlNet $\epsilon$-prediction ([`ddpm.py:915`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L915) & [`cldm.py:376-392`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/cldm.py#L376-L392)):**
     $$\hat{\epsilon}_\theta(x_t(\epsilon), t, c) = \text{ControlledUNet}\big(x_t(\epsilon), t, c_{\text{crossattn}}, \text{ControlNet}(x_t(\epsilon), c_{\text{concat}}, t, c_{\text{crossattn}})\big)$$
  4. **One-step clean-latent prediction $\hat{x}_0(\epsilon)$ ([`ddpm.py:916`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L916) & [`predict_start_from_noise` at `ddpm.py:293-297`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L293-L297)):**
     $$\hat{x}_0(\epsilon) = \frac{1}{\sqrt{\bar{\alpha}_t}} \, x_t(\epsilon) - \sqrt{\frac{1}{\bar{\alpha}_t} - 1} \; \hat{\epsilon}_\theta(x_t(\epsilon), t, c)$$

---

## 5. Mathematical Design of the Three LoRA Versions `(A, B, C)`

For every training step, given a paired sample $(x_0, \hat{x}_{\text{LR}}, c, M_{\text{HR}})$:
1. Sample **one** timestep $t \sim \text{Uniform}(\{0, \dots, T-1\})$ per item in the batch.
2. Draw **two independent Gaussian noise samples** $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, I)$ of shape $(B, 4, 64, 64)$.
3. Compute both clean-latent predictions at the **same** $t$:
   $$\hat{x}_0^{(1)} = \text{predict\_x0\_out\_mix}(x_0, \hat{x}_{\text{LR}}, c, t, \text{noise}=\epsilon_1)$$
   $$\hat{x}_0^{(2)} = \text{predict\_x0\_out\_mix}(x_0, \hat{x}_{\text{LR}}, c, t, \text{noise}=\epsilon_2)$$
4. Convert the HR pixel-space binary text mask $M_{\text{HR}} \in \{0, 1\}^{B \times 1 \times 512 \times 512}$ (built from `readable == yes` boxes `(x, y, w, h)` in `data/text_boxes_*.csv`) to latent resolution $(B, 1, 64, 64)$ via `F.max_pool2d(M_HR, kernel_size=8, stride=8)` so any $8 \times 8$ pixel block touching a text box has $M_{\text{lat}} = 1$.
5. Define the three component losses:
   - **Original 2-noise loss ($\mathcal{L}_{\text{orig}}$):**
     $$\mathcal{L}_{\text{orig}} = \frac{1}{2} \left( \|x_0 - \hat{x}_0^{(1)}\|_1 + \|x_0 - \hat{x}_0^{(2)}\|_1 \right)$$
     where $\|\cdot\|_1$ is the mean absolute error over all $(B, 4, 64, 64)$ elements.
   - **Masked text-region reconstruction loss ($\mathcal{L}_{\text{text-rec}}$):**
     $$\mathcal{L}_{\text{text-rec}} = \frac{1}{2} \left( \frac{\| M_{\text{lat}} \odot (x_0 - \hat{x}_0^{(1)}) \|_1}{\text{mean}(M_{\text{lat}}) + \varepsilon} + \frac{\| M_{\text{lat}} \odot (x_0 - \hat{x}_0^{(2)}) \|_1}{\text{mean}(M_{\text{lat}}) + \varepsilon} \right)$$
   - **Cross-noise text-region consistency loss ($\mathcal{L}_{\text{text-cons}}$):**
     $$\mathcal{L}_{\text{text-cons}} = \frac{\| M_{\text{lat}} \odot (\hat{x}_0^{(1)} - \hat{x}_0^{(2)}) \|_1}{\text{mean}(M_{\text{lat}}) + \varepsilon}$$

### The Three Fair-Comparison LoRA Configurations
**Critical fairness rule:** All three versions (`A`, `B`, and `C`) draw **two independent noise samples $\epsilon_1, \epsilon_2$ per step** at the same timestep $t$, run **two forward passes per step**, and use **both** $\hat{x}_0^{(1)}$ and $\hat{x}_0^{(2)}$ inside the normal loss $\mathcal{L}_{\text{orig}}$ (and inside $\mathcal{L}_{\text{text-rec}}$ for `B` and `C`). This guarantees that `C` does not get more gradient signal or more noise samples than `A` or `B`.

| LoRA Version | Noise Samples / Step | Loss Formula | Purpose in Ablation |
|---|---|---|---|
| **LoRA `(A)`** | **2** ($\epsilon_1, \epsilon_2$) | $\mathcal{L}_A = \mathcal{L}_{\text{orig}} = \frac{1}{2}\big(\|x_0 - \hat{x}_0^{(1)}\|_1 + \|x_0 - \hat{x}_0^{(2)}\|_1\big)$ | Baseline LoRA with original DoSSR loss averaged over **both** noise samples so `(C)` gets no extra gradient signal. |
| **LoRA `(B)`** | **2** ($\epsilon_1, \epsilon_2$) | $\mathcal{L}_B = \mathcal{L}_{\text{orig}} + \lambda_{\text{rec}} \, \mathcal{L}_{\text{text-rec}}$ | Uses **both** noise samples for $\mathcal{L}_{\text{orig}}$ and $\mathcal{L}_{\text{text-rec}}$; tests whether higher text-region weight alone is enough. |
| **LoRA `(C)`** | **2** ($\epsilon_1, \epsilon_2$) | $\mathcal{L}_C = \mathcal{L}_{\text{orig}} + \lambda_{\text{rec}} \, \mathcal{L}_{\text{text-rec}} + \lambda_{\text{cons}} \, \mathcal{L}_{\text{text-cons}}$ | **B vs. C** directly isolates what the cross-noise text-region consistency loss ($\mathcal{L}_{\text{text-cons}}$) adds. |

> [!IMPORTANT]
> **Phase 1 Goal (Next Mentor Report Deliverable):** Before running the full `A / B / C` comparison, **Phase 1 aims at a short trial run of version `C` on Colab Pro with peak VRAM (`peak_allocated_gib`, `peak_reserved_gib`) and step time logged.**

---

## 6. Training, Validation, and Test Data Question (Open Decision — Ask User)

We need an HR dataset with text bounding boxes that **does NOT overlap** the 100 RealSR evaluation images (`RealSRVal_crop128/test_LR` and `test_HR`). We do **not** decide this automatically; the options are listed below for the user and mentor to choose.

### 6.1 What Upstream DoSSR Training Config Expects (`VERIFIED` in `external/DOSSR`)
In [`external/DOSSR/configs/dataset/general_deg_realesrgan_train.yaml:1-62`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/dataset/general_deg_realesrgan_train.yaml#L1-L62) and [`external/DOSSR/dataset/realesrgan.py:114-180`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/realesrgan.py#L114-L180):
- **`meta_info: preset/train_datasets/training_datasets.txt`:** A plain-text list of file paths to HR training images (`preset/train_datasets/training_datasets.txt` is **not** shipped in the public `external/DOSSR` repo).
- **`out_size: 512`, `crop_type: random`, `use_hflip: True`:** Each HR image is randomly cropped to `512x512` on CPU (`realesrgan.py:124-136`) and paired with synthetic blur kernels (`kernel1`, `kernel2`, `sinc_kernel`).
- **On-GPU degradation (`RealESRGANBatchTransform` in `batch_transform.py:136-270`):** Synthesizes $4\times$ degraded LR (`128x128`) from the `512x512` HR patch using second-order blur + resize + Gaussian/Poisson noise + JPEG compression, then bicubic-upsamples LR back to `512x512` (`hint`).
- **No validation data loader in `train_cldm.yaml`:** Upstream `DataModule` ([`dataset/data_module.py:21-40`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/data_module.py#L21-L40)) only sets up `train_dataloader` unless `val_config` is added.

### 6.2 What Our Text-Region LoRA Pipeline Needs per Split
1. **Training split (Disjoint from Test):**
   - Clean HR images (`>= 512x512`) that contain readable text regions, plus word/line bounding boxes `(x, y, w, h)` in HR coordinates.
   - Either paired real `(LR 128x128, HR 512x512, text_boxes)` crops, OR clean `(HR 512x512, text_boxes)` crops degraded by DoSSR's Real-ESRGAN degradation (with `_dequeue_and_enqueue` disabled so each mask stays paired with its image).
2. **Validation split (Disjoint from Train and Test):**
   - Small held-out set (e.g., `10–20` text-containing image pairs) used only to check loss curves, pick $\lambda_{\text{rec}}, \lambda_{\text{cons}}$, and check for overfitting before testing.
3. **Test split (Strictly Held Out — Never Used in Training or Hyperparameter Tuning):**
   - `RealSRVal_crop128` (`test_LR` `128x128` $\rightarrow$ `test_HR` `512x512`, 100 images, labeled by Member 2 in `data/text_boxes_realsr.csv`).
   - If `RealSRVal_crop128` has too few `readable=yes` text boxes, the mentor requested **`Real-CE x4`** for text evaluation.

### 6.3 Candidate Data Options (`TODO: confirm — Ask User to Choose`)
- **Option 1 — `Real-CE` (`Real-CE x4`):**
  - Can provide real-world paired LR-HR images with text/character regions.
  - **`TODO: confirm` (Unknowns we do not guess):**
    1. Is `Real-CE x4` already on the team Drive, or where should we download it?
    2. What are the official train / val / test split sizes and image dimensions in `Real-CE x4`?
    3. Does `Real-CE x4` include ground-truth text bounding boxes and transcripts, or do we need to label boxes with our CSV tool?
- **Option 2 — Public Scene-Text HR Dataset (e.g., ICDAR 2015 / TextOCR / HierText) + DoSSR Real-ESRGAN Degradation:**
  - Uses existing scene-text HR images that already have human-verified word bounding boxes and transcripts, cropped to `512x512` around text regions, with $4\times$ LR (`128x128`) synthesized by DoSSR's own Real-ESRGAN degradation (`batch_transform.py` without `_dequeue_and_enqueue`).
  - Completely disjoint from both `RealSR` and `Real-CE`.
- **Option 3 — Official RealSR Training Split (Filtered for Text Signs, Disjoint from `RealSRVal_crop128`):**
  - Extract `512x512` paired `(LR, HR)` crops containing text from the RealSR **training** camera folders (verifying zero filename/scene overlap with the 100 `RealSRVal_crop128` test images) and annotate text boxes using `make_text_contact_sheets.py`.

---

## 7. Task C.10: LoRA Trial Estimates (`ESTIMATE — NOT TESTED`)

> [!WARNING]
> Every number in the table below is an **engineering estimate (`ESTIMATE — NOT TESTED`)**. None of these numbers have been run on Colab yet.

| Item | Estimate (`ESTIMATE — NOT TESTED`) | Notes / Basis |
|---|---|---|
| **Base Frozen Model Size** | `~1,330M` total parameters (`~969M` SD 2.1 UNet/VAE/OpenCLIP + `~361M` ControlNet) | Upstream `ControlLDM.configure_optimizers` ([`cldm.py:434`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/cldm.py#L434)) trains all `361M` ControlNet parameters when not using LoRA. |
| **LoRA Target Modules** | `to_q`, `to_k`, `to_v`, `to_out.0` in `CrossAttention` layers of `self.control_model` (ControlNet) and `self.model.diffusion_model.output_blocks` (UNet decoder) | Avoids the `with torch.no_grad():` block in `ControlledUnetModel.forward` (`input_blocks` / `middle_block`) without needing to unfreeze the whole UNet. |
| **LoRA Rank & Trainable Parameters (`ESTIMATE`)** | Rank `r = 4`: **`~0.8M – 1.6M`** trainable params (`< 0.15%` of total)<br>Rank `r = 8`: **`~1.6M – 3.2M`** trainable params (`< 0.25%` of total) | Small enough to prevent overfitting on a small text-rich training set and keeps optimizer state under `~25 MiB`. |
| **Patch Size (`ESTIMATE`)** | HR: **`512 × 512`** (`jpg`, `hint`, `M_HR`)<br>LR: **`128 × 128`**<br>Latent (`x`, `preprocess_x`, `pred_x0`, `M_lat`): **`4 × 64 × 64`** (`1 × 64 × 64` for mask) | Matches [`configs/dataset/general_deg_realesrgan_train.yaml:13`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/dataset/general_deg_realesrgan_train.yaml#L13) (`out_size: 512`) and `image_size: 64` in [`cldm_v21.yaml:16`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/model/cldm_v21.yaml#L16). |
| **Batch Size & Grad Accumulation (`ESTIMATE`)** | Micro-batch size **`1` or `2`** per GPU step, with `accumulate_grad_batches = 8` or `4` (effective batch size **`8`**) | Upstream full-ControlNet training used `batch_size: 8` across big GPUs ([`general_deg_realesrgan_train.yaml:57`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/dataset/general_deg_realesrgan_train.yaml#L57)); for single-GPU Colab with 2 noise samples per step, `batch_size = 1` or `2` is safest. |
| **Peak VRAM (`ESTIMATE`)** | **`~12 – 18 GiB`** (`batch_size = 1`, FP16 autocast, `use_checkpoint: True`, 2 noise samples)<br>**`~16 – 24 GiB`** (`batch_size = 2`, FP16 autocast, `use_checkpoint: True`, 2 noise samples) | Inference uses `7.32 GiB` allocated / `8.29 GiB` reserved (`VERIFIED`). Training keeps gradients for `ControlNet` + `UNet output_blocks` (with activation checkpointing enabled at [`cldm_v21.yaml:28, 61`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/model/cldm_v21.yaml#L28)). Note: we can backward on $\epsilon_1$ and $\epsilon_2$ sequentially (or keep $\hat{x}_0^{(1)}.\text{detach}()$ as target in a symmetric stop-grad or concatenate batch `2*B`) to bound peak activation memory. Fits easily on **Colab Pro L4 (`22.5 GiB`) / A100 (`80 GiB`)**. |
| **Trial Step Time (`ESTIMATE`)** | **`~0.8 – 1.8 s / step`** (`batch_size = 1`, 2 noise samples per step on A100/L4) | 100-step trial run should take **`~2 – 4 minutes`** (`ESTIMATE — NOT TESTED`) after checkpoint loading. |

### What Every LoRA Trial Run Must Log
1. **Run configuration (`lora_run_manifest.json`):**
   - `lora_variant`: `"A"`, `"B"`, or `"C"`
   - `lora_rank`, `lora_alpha`, `target_modules`, `trainable_param_count`, `total_param_count`
   - `prompt_mode` and exact `prompt_string`
   - `batch_size`, `accumulate_grad_batches`, `lr`, `max_steps`, `seed`
   - `lambda_rec`, `lambda_cons`, `noise_samples_per_step` (`2`)
   - `sdpa_enabled`, `precision` (`"fp16"`), `gpu_name`
2. **Per-step training metrics (`lora_train_log.csv`):**
   - `step`, `t_sampled`
   - `loss_total`, `loss_orig_2noise`, `loss_text_rec`, `loss_text_cons`
   - `mask_active_ratio` (share of $64 \times 64$ latent cells inside text boxes)
   - `step_time_s`, `peak_allocated_gib`, `peak_reserved_gib` (`torch.cuda.max_memory_allocated()`, `torch.cuda.max_memory_reserved()`)
