# Handoff Note for Member 3 — LoRA & Text-Region Loss Lead

**Branch:** `feat/lora-loss`  
**Hardware needed right now:** Any laptop (Windows or Mac) to read the audit and design the LoRA config. Member 1 will run the GPU trial on Colab Pro.  
**Status of audit:** Read-only audit in [`docs/lora_feasibility_and_loss_design.md`](file:///Users/lemon4life/Projects/SC203/docs/lora_feasibility_and_loss_design.md) is **complete (`VERIFIED` in `external/DOSSR`)**. Trial numbers are **`ESTIMATE — NOT TESTED`**.

---

## 1. Team Git & Communication Rules

1. **Work on your own branch:** `feat/lora-loss`.
2. **Only edit your own files:**
   - `docs/lora_feasibility_and_loss_design.md`
   - `docs/handoff_member3.md`
   - Future files when approved: `src/lora_text_loss.py`, `configs/lora_*.yaml`
3. **Do NOT edit `context.md` or anything inside `external/DOSSR/`.** Report your decisions and updates in the team chat. Member 1 will update `context.md` and merge your Pull Request (PR).

---

## 2. What Is Blocked vs. What You Can Do Right Now

- **Blocked by Member 2 (`data/text_boxes_realsr.csv`):**
  - Training on real ground-truth text masks requires Member 2's CSV (`image,box_id,x,y,w,h,transcript,readable`).
- **What you can do right now (Not Blocked):**
  1. Read [`docs/lora_feasibility_and_loss_design.md`](file:///Users/lemon4life/Projects/SC203/docs/lora_feasibility_and_loss_design.md) (Tasks C.9 & C.10 report).
  2. Inspect the upstream training files in `external/DOSSR` (read-only).
  3. Decide the hyperparameters and plan for the three LoRA versions `(A)`, `(B)`, and `(C)` and prototype the loss with a synthetic dummy mask `(B, 1, 64, 64)`.

---

## 3. Key Findings You Need from Tasks C.9 & C.10

Read [`docs/lora_feasibility_and_loss_design.md`](file:///Users/lemon4life/Projects/SC203/docs/lora_feasibility_and_loss_design.md) first. Here is the short summary:

1. **Full training entry point exists in `external/DOSSR`:**
   - [`external/DOSSR/scripts/train.py:10-30`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/scripts/train.py#L10-L30) and [`external/DOSSR/configs/train_cldm.yaml:1-47`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/train_cldm.yaml#L1-L47).
2. **Training prompt vs. Inference prompt:**
   - Upstream training ([`dataset/realesrgan.py:179`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/realesrgan.py#L179)) always passes `"txt": ""` (empty prompt).
   - Upstream inference ([`model/shift_sampler.py:322`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/shift_sampler.py#L322)) passes `"clean, high-resolution, 8k"`.
   - **Rule for LoRA `(A, B, C)`:** All three versions must use the **exact same prompt** at both train time and test time so there is no train/test mismatch and no unfair difference between `A`, `B`, and `C`.
3. **Where the loss and clean-latent prediction $\hat{x}_0$ live:**
   - [`ControlLDM.shared_step` (`model/cldm.py:346-356`)](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/cldm.py#L346-L356) calls `pred_x0 = self.predict_x0_out_mix(x, preprocess_x, c, t, **kwargs)` and computes `loss_preprocess_diff = (x - pred_x0).abs().mean()`.
   - [`LatentDiffusion.predict_x0_out_mix` (`ldm/models/diffusion/ddpm.py:909-917`)](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L909-L917) already takes an optional `noise=` tensor!
   - To draw two independent noise samples $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, I)$ for the same $(x, \text{preprocess\_x}, c, t)$, we simply call `predict_x0_out_mix` twice (`noise=eps_1` and `noise=eps_2`).
4. **Two critical code traps we found in `external/DOSSR`:**
   - **Trap 1 (`batch_transform.py:98-134`):** `RealESRGANBatchTransform._dequeue_and_enqueue` shuffles a 256-image GPU queue (`torch.randperm`) without shuffling extra batch keys like a text mask. Our LoRA data loader must **not** call `_dequeue_and_enqueue` unless the mask is queued with identical indices (or better: use a direct paired dataset without the queue).
   - **Trap 2 (`model/cldm.py:27-35`):** `ControlledUnetModel.forward` wraps `self.input_blocks` and `self.middle_block` in `with torch.no_grad():`. Put LoRA adapters in `self.control_model` (ControlNet) and `diffusion_model.output_blocks` (or override `ControlledUnetModel.forward` in our wrapper).

---

## 4. How the Shared Text-Box CSV Connects to Your Mask

Member 2 produces `data/text_boxes_realsr.csv` with header:
```csv
image,box_id,x,y,w,h,transcript,readable
```
- Coordinates `(x, y, w, h)` are in `512x512` HR pixel space.
- For each image, union all boxes with `readable == yes` into a binary mask $M_{\text{HR}} \in \{0, 1\}^{1 \times 512 \times 512}$.
- Downsample $M_{\text{HR}}$ by $8\times$ to latent size $(1, 64, 64)$ using `F.max_pool2d(M_HR, kernel_size=8, stride=8)`.

---

## 5. What to Do First & What to Send Back to Member 1

1. Clone the repo and create your branch:
   ```bash
   git clone https://github.com/lemon4life/SC203
   cd SC203
   git checkout feat/baseline-runner
   git checkout -b feat/lora-loss
   ```
2. Read [`docs/lora_feasibility_and_loss_design.md`](file:///Users/lemon4life/Projects/SC203/docs/lora_feasibility_and_loss_design.md).
3. Send back to Member 1 in chat:
   - Your confirmation of the loss weights ($\lambda_{\text{rec}}$ for LoRA `B` and `C`, $\lambda_{\text{cons}}$ for LoRA `C`).
   - Your choice of LoRA rank (`r = 4` vs `r = 8`) and target modules (`ControlNet` only vs `ControlNet + UNet output_blocks`).
   - Which dataset we will use for the trial LoRA run on Colab Pro (so we never train on test images).
