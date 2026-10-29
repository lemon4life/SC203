#!/usr/bin/env python3
"""Launch upstream DoSSR inference with its broken local OpenCLIP path bypassed.

DoSSR hard-codes an author-machine checkpoint path when constructing OpenCLIP.
The full DoSSR checkpoint is loaded strictly immediately after model creation,
so this launcher builds the architecture without separate pretrained weights;
the strict DoSSR checkpoint load then supplies all model parameters.
"""

from __future__ import annotations

import argparse
import gc
import runpy
import sys
import time
from pathlib import Path

AUTHOR_OPENCLIP_CHECKPOINT = Path(
    "/group/ossmodelzoo/qinpecui/downloads/open_clip_pytorch_model.bin"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-script", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default=None)
    parser.add_argument("--fp16", action="store_true", help="Enable CUDA float16 autocast")
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Log CUDA allocator snapshots around checkpoint loading and sampling",
    )
    parser.add_argument(
        "--offload-condition-encoder",
        action="store_true",
        help=(
            "Move DoSSR's condition encoder off cuda:0; use cuda:1 when available, "
            "otherwise use the CPU fallback"
        ),
    )
    args, inference_args = parser.parse_known_args()
    if args.fp16 and args.precision == "fp32":
        parser.error("Use either --fp16 or --precision, not conflicting values")
    precision = "fp16" if args.fp16 else (args.precision or "fp32")
    if inference_args[:1] == ["--"]:
        inference_args = inference_args[1:]

    inference_script = args.inference_script.expanduser().resolve()
    if not inference_script.is_file():
        parser.error(f"DoSSR inference script not found: {inference_script}")

    import open_clip
    import torch
    import utils.common as dossr_common
    from ldm import xformers_state

    if not xformers_state.is_xformers_available():
        from ldm.modules import attention as attention_module

        from attention_compat import patch_cross_attention

        if patch_cross_attention(attention_module):
            print(
                "DoSSR compatibility: using PyTorch scaled_dot_product_attention "
                "for vanilla CrossAttention; original projections, heads, scale, "
                "and mask semantics are retained.",
                flush=True,
            )
        else:
            print(
                "DoSSR compatibility: PyTorch SDPA is unavailable; retaining "
                "upstream einsum CrossAttention.",
                flush=True,
            )

    def report_cuda_memory(label: str) -> None:
        if not args.profile_memory:
            return
        if not torch.cuda.is_available():
            print(f"DoSSR memory profile | {label} | CUDA unavailable", flush=True)
            return
        try:
            torch.cuda.synchronize()
        except RuntimeError as error:
            print(
                f"DoSSR memory profile | {label} | CUDA synchronize reported: {error}",
                flush=True,
            )
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        stats = torch.cuda.memory_stats()
        print(
            "DoSSR memory profile | "
            f"{label} | allocated={allocated / 2**30:.2f} GiB | "
            f"reserved={reserved / 2**30:.2f} GiB | "
            f"allocator_cache={max(0, reserved - allocated) / 2**30:.2f} GiB | "
            f"peak_allocated={peak_allocated / 2**30:.2f} GiB | "
            f"peak_reserved={peak_reserved / 2**30:.2f} GiB | "
            f"inactive_split={stats.get('inactive_split_bytes.all.current', 0) / 2**30:.2f} GiB | "
            f"alloc_retries={stats.get('num_alloc_retries', 0)} | "
            f"ooms={stats.get('num_ooms', 0)}",
            flush=True,
        )

    original_create = open_clip.create_model_and_transforms
    original_torch_load = torch.load
    original_load_state_dict = dossr_common.load_state_dict

    def create_without_missing_author_weights(*call_args, **call_kwargs):
        requested = call_kwargs.get("pretrained")
        if requested == str(AUTHOR_OPENCLIP_CHECKPOINT) and not AUTHOR_OPENCLIP_CHECKPOINT.is_file():
            print(
                "DoSSR compatibility: skipping its missing author-local OpenCLIP "
                "initialization file; the subsequent strict DoSSR checkpoint load "
                "provides the model weights."
            )
            call_kwargs["pretrained"] = None
        return original_create(*call_args, **call_kwargs)

    open_clip.create_model_and_transforms = create_without_missing_author_weights

    def torch_load_with_mmap(*load_args, **load_kwargs):
        """Prefer file-backed checkpoint tensors and retry only if mmap is unsupported."""
        gc.collect()
        should_retry_without_mmap = False
        if "mmap" not in load_kwargs:
            load_kwargs["mmap"] = True

        if load_kwargs["mmap"]:
            print("DoSSR compatibility: loading checkpoint with torch.load(mmap=True).", flush=True)

        try:
            try:
                started = time.monotonic()
                result = original_torch_load(*load_args, **load_kwargs)
                print(
                    "DoSSR compatibility: checkpoint deserialization returned "
                    f"after {time.monotonic() - started:.1f}s.",
                    flush=True,
                )
                return result
            except (RuntimeError, TypeError) as error:
                message = str(error).lower()
                if isinstance(error, TypeError):
                    should_retry_without_mmap = (
                        "mmap" in message and "unexpected keyword" in message
                    )
                else:
                    should_retry_without_mmap = (
                        "mmap can only be used" in message
                        or ("mmap" in message and "not supported" in message)
                    )
                if not load_kwargs.get("mmap") or not should_retry_without_mmap:
                    raise

                print(
                    "DoSSR compatibility: mmap is unsupported for this load; "
                    "retrying with mmap=False.",
                    flush=True,
                )
                load_kwargs["mmap"] = False
                gc.collect()
                started = time.monotonic()
                result = original_torch_load(*load_args, **load_kwargs)
                print(
                    "DoSSR compatibility: checkpoint deserialization returned "
                    f"after {time.monotonic() - started:.1f}s with mmap disabled.",
                    flush=True,
                )
                return result
        finally:
            gc.collect()

    def load_state_dict_with_progress(*load_args, **load_kwargs):
        print("DoSSR compatibility: strict model state-dict load starting.", flush=True)
        report_cuda_memory("before strict state-dict load")
        started = time.monotonic()
        result = original_load_state_dict(*load_args, **load_kwargs)
        print(
            "DoSSR compatibility: strict model state-dict load finished "
            f"after {time.monotonic() - started:.1f}s.",
            flush=True,
        )
        report_cuda_memory("after strict state-dict load")
        return result

    torch.load = torch_load_with_mmap
    dossr_common.load_state_dict = load_state_dict_with_progress

    if args.profile_memory:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        report_cuda_memory("before model creation")
        from model.cldm import ControlLDM
        from model.shift_sampler import SpacedSampler

        original_model_to = ControlLDM.to

        def model_to_with_memory_profile(model, *to_args, **to_kwargs):
            report_cuda_memory("before model device transfer")
            started = time.monotonic()
            result = original_model_to(model, *to_args, **to_kwargs)
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except RuntimeError as error:
                    print(
                        "DoSSR model device-transfer profile | CUDA synchronize reported: "
                        f"{error}",
                        flush=True,
                    )
            print(
                "DoSSR model device-transfer profile | elapsed="
                f"{time.monotonic() - started:.2f}s",
                flush=True,
            )
            report_cuda_memory("after model device transfer")
            return result

        ControlLDM.to = model_to_with_memory_profile
        original_sample = SpacedSampler.shift_sample_3_order

        def sample_with_memory_profile(sampler, *sample_args, **sample_kwargs):
            report_cuda_memory("before sampler")
            started = time.monotonic()
            try:
                return original_sample(sampler, *sample_args, **sample_kwargs)
            finally:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except RuntimeError as error:
                        print(
                            "DoSSR sampler profile | CUDA synchronize reported: "
                            f"{error}",
                            flush=True,
                        )
                print(
                    "DoSSR sampler profile | elapsed="
                    f"{time.monotonic() - started:.2f}s",
                    flush=True,
                )
                report_cuda_memory("after sampler")

        SpacedSampler.shift_sample_3_order = sample_with_memory_profile

    if args.offload_condition_encoder:
        if not torch.cuda.is_available():
            raise RuntimeError("--offload-condition-encoder requires an available CUDA device")

        # DoSSR's condition encoder is a separate copy of the first-stage encoder.
        # Prefer a second CUDA device so both weights and activations stay in VRAM;
        # retain CPU as the single-GPU fallback. Only the latent returns to cuda:0.
        from model.cldm import ControlLDM

        original_apply_condition_encoder = ControlLDM.apply_condition_encoder
        condition_device = (
            torch.device("cuda:1")
            if torch.cuda.device_count() > 1
            else torch.device("cpu")
        )
        if condition_device.type == "cuda":
            print(
                "DoSSR compatibility: condition encoder offload enabled on "
                f"{condition_device} ({torch.cuda.get_device_name(condition_device)}); "
                "encoded latents return to the main model device.",
                flush=True,
            )
        else:
            print(
                "DoSSR compatibility: only one CUDA device is visible; "
                "using the existing CPU condition-encoder fallback.",
                flush=True,
            )

        def apply_condition_encoder_with_offload(model, control):
            destination = control.device
            model.cond_encoder.to(condition_device)
            if condition_device.type == "cpu":
                torch.cuda.empty_cache()
                gc.collect()
                latent = original_apply_condition_encoder(model, control.to("cpu"))
            else:
                with torch.cuda.device(condition_device):
                    latent = original_apply_condition_encoder(
                        model, control.to(condition_device)
                    )
            if precision == "fp16":
                latent = latent.to(dtype=torch.float16)
            return latent.to(destination)

        ControlLDM.apply_condition_encoder = apply_condition_encoder_with_offload

    sys.argv = [str(inference_script), *inference_args]
    if precision == "fp16":
        if not torch.cuda.is_available():
            raise RuntimeError("precision=fp16 requires an available CUDA device")
        print(
            "DoSSR compatibility: enabling CUDA float16 autocast for inference.",
            flush=True,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            cache_enabled=False,
        ):
            runpy.run_path(str(inference_script), run_name="__main__")
    else:
        runpy.run_path(str(inference_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
