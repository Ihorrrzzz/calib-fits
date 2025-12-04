# calibration/mkmasterbias.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval

from calib_config import CalibConfig


def _get_image_type(header, cfg: CalibConfig) -> str:
    key = cfg.get("HEADER_SPECIFICATION", "image_type_keyword", "IMAGETYP")
    return str(header.get(key, "")).strip().upper()


def _get_bias_label(cfg: CalibConfig) -> str:
    val = cfg.get("BIAS_SUBTRACTION", "bias_keyword")
    if val is not None:
        return str(val).strip().upper()
    # fallback if someone puts it in HEADER_SPECIFICATION
    val = cfg.get("HEADER_SPECIFICATION", "bias_keyword")
    if val is not None:
        return str(val).strip().upper()
    return "BIAS"


def find_bias_frames(
    files: List[str],
    cfg: CalibConfig,
    verbose: bool = False,
) -> List[str]:
    bias_label = _get_bias_label(cfg)
    result: List[str] = []

    for fname in files:
        try:
            with fits.open(fname, memmap=True) as hdul:
                header = hdul[0].header
                imagetyp = _get_image_type(header, cfg)
                if imagetyp == bias_label:
                    result.append(fname)
        except Exception as e:
            if verbose:
                print(f"[mkmasterbias] Skipping {fname}: {e}")

    if verbose:
        print(f"[mkmasterbias] Found {len(result)} bias frames (label = {bias_label}).")
    return sorted(result)


def make_png(fits_path: str | Path, verbose: bool = False) -> str:
    from matplotlib import pyplot as plt

    fits_path = Path(fits_path)
    png_path = fits_path.with_suffix(".png")

    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[0].data

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    plt.figure()
    plt.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    plt.colorbar(label="ADU (zscale)")
    plt.title("Master Bias")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    if verbose:
        print(f"[mkmasterbias] PNG preview saved to {png_path}")

    return str(png_path)


def create_master_bias(
    all_files: List[str],
    cfg: CalibConfig,
    output_filename: str = "masterbias.fits",
    method: Optional[str] = None,
    sigma: Optional[float] = None,
    make_png: bool = False,
    verbose: bool = False,
) -> str:
    """
    Create a master bias frame from all_files, using config-driven settings.
    Returns the full path to the saved master bias FITS file.
    """
    bias_enabled = cfg.get("IMAGE_PROCESSING", "bias_subtraction")
    if bias_enabled is False:
        if verbose:
            print("[mkmasterbias] Bias subtraction disabled in config; not creating master bias.")
        working_dir = Path(cfg.get("DATA_STRUCTURE", "working_dir", "./work"))
        working_dir.mkdir(parents=True, exist_ok=True)
        return str(working_dir / output_filename)

    full_config = cfg.config
    method_cfg = (
        method
        or full_config.get("IMAGE_PROCESSING", {}).get("bias_subtraction_method", "MedianSigmaClipped")
    )
    sigma_cfg = float(
        sigma
        if sigma is not None
        else full_config.get("IMAGE_PROCESSING", {}).get("bias_subtraction_sigma", 2.5)
    )

    working_dir = Path(cfg.get("DATA_STRUCTURE", "working_dir", "./work"))
    working_dir.mkdir(parents=True, exist_ok=True)
    out_path = working_dir / output_filename

    bias_files = find_bias_frames(all_files, cfg, verbose=verbose)
    if not bias_files:
        raise RuntimeError("[mkmasterbias] No bias frames found; cannot create master bias.")

    if verbose:
        print(f"[mkmasterbias] Combining {len(bias_files)} bias frames using {method_cfg} (sigma={sigma_cfg}).")

    stack_list = []
    for fname in bias_files:
        with fits.open(fname, memmap=True) as hdul:
            data = hdul[0].data.astype(np.float32)
        stack_list.append(data)

    stack = np.stack(stack_list, axis=0)

    method_lower = str(method_cfg).lower()
    if method_lower == "mediansigmaclipped":
        clipped = sigma_clip(stack, sigma=sigma_cfg, axis=0)
        master = np.nanmedian(clipped, axis=0)
    elif method_lower == "median":
        master = np.median(stack, axis=0)
    elif method_lower in {"average", "mean"}:
        master = np.mean(stack, axis=0)
    else:
        if verbose:
            print(f"[mkmasterbias] Unknown method '{method_cfg}', using sigma-clipped median as fallback.")
        clipped = sigma_clip(stack, sigma=sigma_cfg, axis=0)
        master = np.nanmedian(clipped, axis=0)

    hdu = fits.PrimaryHDU(master.astype(np.float32))
    hdr = hdu.header
    hdr["HISTORY"] = "Master bias created by mkmasterbias.py"
    hdr["MB_NFRM"] = (len(bias_files), "Number of bias frames")
    hdr["MB_METH"] = (str(method_cfg), "Bias combination method")
    hdr["MB_SIG"] = (float(sigma_cfg), "Sigma for sigma-clipping")

    hdu.writeto(out_path, overwrite=True)

    if verbose:
        print(f"[mkmasterbias] Master bias saved to {out_path}")

    if make_png:
        make_png(out_path, verbose=verbose)

    return str(out_path)


# ----------------------------------------------------------------------
# CLI for standalone use
# ----------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(description="Create master bias from list of FITS files.")
    parser.add_argument(
        "-l",
        "--list",
        required=True,
        help="Text file with FITS paths (one per line).",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to LISNYKY_Moravian-C4-16000.ini",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="masterbias.fits",
        help="Output master bias filename (inside working_dir).",
    )
    parser.add_argument(
        "-m",
        "--method",
        help="Override bias combination method (config: IMAGE_PROCESSING.bias_subtraction_method).",
    )
    parser.add_argument(
        "-s",
        "--sigma",
        type=float,
        help="Override sigma for sigma-clipping.",
    )
    parser.add_argument(
        "-p",
        "--png",
        action="store_true",
        help="Create PNG preview of the master bias.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )

    args = parser.parse_args()
    cfg = CalibConfig(args.config)

    with open(args.list, "r", encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]

    create_master_bias(
        files,
        cfg,
        output_filename=args.output,
        method=args.method,
        sigma=args.sigma,
        make_png=args.png,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    _cli()
