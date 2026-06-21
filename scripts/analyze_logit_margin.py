from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import torch

from constants import ANALYZE_MARGIN_SAMPLES
from preferences.analyze_logit_margin import analyze_logit_margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=ANALYZE_MARGIN_SAMPLES)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyze_logit_margin(device=device, n_samples=args.n)


if __name__ == "__main__":
    main()
