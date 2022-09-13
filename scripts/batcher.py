"""Batch-mode driver for txt2img.py."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import itertools
import os
import random
import regex
import yaml
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable

import transformers

from txt2img import init_model, generate_image


def read_batch_file(batch_file: str) -> dict:
    """Read batch file."""
    with open(batch_file, "r", encoding="utf-8") as f:
        batch = yaml.safe_load(f)
    return batch


def parse_prompt(prompt: str) -> Iterable[str]:
    """
    Parse the prompt.

    We generate prompts for each value in {} separated by a |.
    """
    # Break the list into a flat list of strings: the fixed and variable parts
    parts = list(
        itertools.chain.from_iterable(
            regex.split(r"\s*}\s*", sub) for sub in regex.split(r"\s*{\s*", prompt)
        )
    )

    # Break the variable parts into lists; now the fixed parts are lists of
    # single strings and the variable parts are lists of strings, one per
    # variant.
    opts = (regex.split(r"\s*\|\s*", sub) for sub in parts)

    # Build the cross product of the lists
    prompts = itertools.product(*opts)

    # Join the sub-lists into strings: now we have a list of prompts covering
    # all the variants.
    return (" ".join(sub) for sub in prompts)


def main() -> None:
    """Execute main function."""
    parser = ArgumentParser(__doc__)

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    parser.add_argument(
        type=str,
        help="batch description yaml file",
        dest="batch_file",
    )

    args = parser.parse_args()

    # Hardcode these
    args.laion400m = False

    # Read batch file
    batch = read_batch_file(args.batch_file)

    # Prevent nuisance warnings when the model is loaded.
    transformers.logging.set_verbosity_error()

    # Initialize model
    device, precision_scope, model = init_model(args)

    # Compute the base path
    base_path = os.path.join("outputs", "batches")
    os.makedirs(base_path, exist_ok=True)
    base_count = len(os.listdir(base_path))
    base_path = os.path.join(base_path, f"batch-{base_count:05d}")

    # Process batch
    for i, prompt in enumerate(batch["prompts"]):
        batch_args = Namespace()

        # Retrieve prompt; error and continue if not found
        if "prompt" in prompt:
            subprompts = parse_prompt(prompt["prompt"])
        else:
            print(f"Prompt not found for prompt {i+1}; continuing to the next prompt.")
            continue

        print(f"Processing prompt {i+1} of {len(batch['prompts'])}")

        # Output directory; drefaults to "outputs/batch-samples"
        if "dir" in prompt:
            batch_args.outdir = os.path.join(base_path, prompt["dir"])
        else:
            batch_args.outdir = os.path.join(base_path, f"prompt-{i+1:05d}")

        for j, subprompt in enumerate(subprompts):
            batch_args.prompt = subprompt

            # Height as H; default to 512
            if "height" in prompt:
                batch_args.H = prompt["height"]
            else:
                batch_args.H = 512

            # Width as W; default to 512
            if "width" in prompt:
                batch_args.W = prompt["width"]
            else:
                batch_args.W = 512

            # Latent channels as C; default to 4
            if "latent_channels" in prompt:
                batch_args.C = prompt["latent_channels"]
            else:
                batch_args.C = 4

            # Downsample factor as f; default to 8
            if "downsample_factor" in prompt:
                batch_args.f = prompt["downsample_factor"]
            else:
                batch_args.f = 8

            # classifer-free guidance scale as scale; default to 7.5
            if "cfg-scale" in prompt:
                batch_args.scale = prompt["cfg-scale"]
            else:
                batch_args.scale = 7.5

            # Seed; default to random
            if "seed" in prompt:
                batch_args.seed = prompt["seed"]
            else:
                batch_args.seed = random.randint(0, 2**32 - 1)

            # Samples as n_iter; defaults to 1
            if "samples" in prompt:
                batch_args.n_iter = prompt["samples"]
            else:
                batch_args.n_iter = 1

            # Steps as ddim_steps; defaults to 50
            if "steps" in prompt:
                batch_args.ddim_steps = prompt["steps"]
            else:
                batch_args.ddim_steps = 50

            # Print the namespace to a yaml file
            os.makedirs(batch_args.outdir, exist_ok=True)
            with open(
                os.path.join(batch_args.outdir, f"prompt-{j:05d}.yaml"),
                "w",
                encoding="utf-8",
            ) as f:
                yaml.dump(batch_args, f)

            # We don't want a grid; we'll worry about making our own
            batch_args.skip_grid = True

            # We don't want the plms sampler for now
            batch_args.plms = False

            # Trying to run multiple samples in a batch doesn't work with our
            # hardware.
            batch_args.n_samples = 1

            # We ignore this parameter
            batch_args.n_rows = 1

            # We won't read prompts from a file
            batch_args.from_file = False

            # No fixed code
            batch_args.fixed_code = False

            # Don't skip saving
            batch_args.skip_save = False

            # Hard-code ddim_eta for now to 0.0
            batch_args.ddim_eta = 0.0

            print(f"Processing subprompt {j+1} of prompt {i+1}.")
            generate_image(
                model=model,
                device=device,
                precision_scope=precision_scope,
                opt=batch_args,
            )

    print(f"Batch processing complete; output in {batch_args.outdir}.")


if __name__ == "__main__":
    main()
