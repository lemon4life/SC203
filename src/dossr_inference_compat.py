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
        "--offload-condition-encoder",
        action="store_true",
        help="Run DoSSR's condition encoder on CPU to reduce CUDA activation memory",
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
        started = time.monotonic()
        result = original_load_state_dict(*load_args, **load_kwargs)
        print(
            "DoSSR compatibility: strict model state-dict load finished "
            f"after {time.monotonic() - started:.1f}s.",
            flush=True,
        )
        return result

    torch.load = torch_load_with_mmap
    dossr_common.load_state_dict = load_state_dict_with_progress

    if args.offload_condition_encoder:
        if not torch.cuda.is_available():
            raise RuntimeError("--offload-condition-encoder requires an available CUDA device")

        # DoSSR's condition encoder is a separate copy of the first-stage encoder.
        # Keep its weights and activations on CPU, then move only the small latent
        # back to the model device. This avoids changing upstream files.
        from model.cldm import ControlLDM

        original_apply_condition_encoder = ControlLDM.apply_condition_encoder

        def apply_condition_encoder_on_cpu(model, control):
            destination = control.device
            model.cond_encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            latent = original_apply_condition_encoder(model, control.to("cpu"))
            if precision == "fp16":
                latent = latent.to(dtype=torch.float16)
            return latent.to(destination)

        ControlLDM.apply_condition_encoder = apply_condition_encoder_on_cpu
        print(
            "DoSSR compatibility: condition encoder CPU offload enabled; "
            "its latent will return to the model device.",
            flush=True,
        )

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
