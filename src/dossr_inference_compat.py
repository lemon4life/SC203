#!/usr/bin/env python3
"""Launch upstream DoSSR inference with its broken local OpenCLIP path bypassed.

DoSSR hard-codes an author-machine checkpoint path when constructing OpenCLIP.
The full DoSSR checkpoint is loaded strictly immediately after model creation,
so this launcher builds the architecture without separate pretrained weights;
the strict DoSSR checkpoint load then supplies all model parameters.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

AUTHOR_OPENCLIP_CHECKPOINT = Path(
    "/group/ossmodelzoo/qinpecui/downloads/open_clip_pytorch_model.bin"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-script", type=Path, required=True)
    args, inference_args = parser.parse_known_args()
    if inference_args[:1] == ["--"]:
        inference_args = inference_args[1:]

    inference_script = args.inference_script.expanduser().resolve()
    if not inference_script.is_file():
        parser.error(f"DoSSR inference script not found: {inference_script}")

    import open_clip

    original_create = open_clip.create_model_and_transforms

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
    sys.argv = [str(inference_script), *inference_args]
    runpy.run_path(str(inference_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
