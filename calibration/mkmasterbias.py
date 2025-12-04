# calibration/mkmasterbias.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt

try:
    from .calib_config import CalibConfig
except ImportError:
    from calibration.calib_config import CalibConfig


def find_bias_frames(files: List[str], cfg: CalibConfig, verbose: bool = False) -> List[str]:
    """
    Select bias frames using HEADER_SPECIFICATION.image_type_keyword
    and BIAS_SUBTRACTION.bias_keyword.
    """
    image_type_key = cfg.get("HEADER_SPECIFICATION", "image_type_keyword", "IMAGETYP")
    expected_bias = str(cfg.get("BIAS_SUBTRACTION", "bias_keyword", "BIAS")).strip().upper()

    bias_files: List[str] = []
    for fname in files:
        try:
            with fits.open(fname, memmap=False) as hdul:
                hdr = hdul[0].header
                imagetyp = str(hdr.get(image_type_key, "")).strip().upper()
                if imagetyp == expected_bias:
                    bias_files.append(fname)
        except Exception as e:
            if verbose:
                print(f"[mkmasterbias] Skipping {fname}: {e}")
    if verbose:
        print(f"[mkmasterbias] Found {len(bias_files)} bias frames.")
    return sorted(bias_files)


def make_png(fits_path: Path, png_path: Path, verbose: bool = False) -> None:
    """Create a PNG preview of the master bias."""
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    plt.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
    plt.colorbar(label="Pixel value (zscale)")
    plt.title("Master Bias")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[mkmasterbias] PNG saved to {png_path}")


def create_master_bias(
    files: List[str],
    cfg: CalibConfig,
    output_filename: str = "masterbias.fits",
    method: Optional[str] = None,
    sigma: Optional[float] = None,
    make_png_flag: bool = False,
    verbose: bool = False,
) -> str:
    """
    Create master bias using config options.

    Returns full path to created masterbias FITS.
    """
    full_cfg = cfg.config
    working_dir = Path(cfg.get("DATA_STRUCTURE", "working_dir", "./work"))
    results_aux_dir = Path(cfg.get("DATA_STRUCTURE", "results_aux_dir", "./results/aux"))
    working_dir.mkdir(parents=True, exist_ok=True)
    results_aux_dir.mkdir(parents=True, exist_ok=True)

    bias_enabled = full_cfg.get("IMAGE_PROCESSING", {}).get("bias_subtraction", True)
    if not bias_enabled:
        if verbose:
            print("[mkmasterbias] Bias subtraction disabled in config; not creating master bias.")
        return str(working_dir / output_filename)

    # Choose method & sigma:
    method_cfg = (
        method
        or full_cfg.get("IMAGE_PROCESSING", {}).get("bias_subtraction_method")
        or full_cfg.get("BIAS_SUBTRACTION", {}).get("option")
        or "MedianSigmaClipped"
    )
    sigma_cfg = float(
        sigma
        if sigma is not None
        else full_cfg.get("IMAGE_PROCESSING", {}).get("bias_subtraction_sigma", 2.5)
    )

    bias_files = find_bias_frames(files, cfg, verbose=verbose)
    if not bias_files:
        raise RuntimeError("No bias frames found to create master bias.")

    if verbose:
        print(f"[mkmasterbias] Combining {len(bias_files)} bias frames with {method_cfg}, sigma={sigma_cfg}")

    stack_list = []
    for fname in bias_files:
        with fits.open(fname, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float32)
            stack_list.append(data)

    bias_stack = np.stack(stack_list, axis=0)

    if method_cfg == "MedianSigmaClipped":
        clipped = sigma_clip(bias_stack, sigma=sigma_cfg, axis=0)
        master_bias = np.nanmedian(clipped, axis=0)
    elif method_cfg.lower() in ("median", "med"):
        master_bias = np.median(bias_stack, axis=0)
    elif method_cfg.lower() in ("average", "mean"):
        master_bias = np.mean(bias_stack, axis=0)
    else:
        if verbose:
            print(f"[mkmasterbias] Unsupported method '{method_cfg}', falling back to median.")
        master_bias = np.median(bias_stack, axis=0)

    out_path = working_dir / output_filename
    hdu = fits.PrimaryHDU(master_bias.astype(np.float32))
    hdu.header["MB_COMB"] = (method_cfg, "Master bias combination method")
    hdu.writeto(out_path, overwrite=True)

    # Optionally mirror to results/aux so it can be used as a library file later
    aux_path = results_aux_dir / output_filename
    aux_path.write_bytes(out_path.read_bytes())

    if verbose:
        print(f"[mkmasterbias] Master bias written to {out_path}")
        print(f"[mkmasterbias] Copy stored to {aux_path}")

    if make_png_flag:
        png_path = out_path.with_suffix(".png")
        make_png(out_path, png_path, verbose=verbose)

    return str(out_path)


def _read_list_file(list_path: Path) -> List[str]:
    with list_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create master bias frame from bias images.")
    parser.add_argument("-l", "--list", required=True, help="Text file with list of FITS files (one per line)")
    parser.add_argument("-o", "--output", default="masterbias.fits", help="Output master bias filename")
    parser.add_argument("-m", "--method", help="Override bias subtraction method")
    parser.add_argument("-s", "--sigma", type=float, help="Override sigma for sigma-clipping")
    parser.add_argument("-c", "--config", help="Path to config .ini file", default="config.ini")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-p", "--png", action="store_true", help="Also create PNG preview")

    args = parser.parse_args(argv)

    cfg = CalibConfig(args.config)
    files = _read_list_file(Path(args.list))

    try:
        out_path = create_master_bias(
            files=files,
            cfg=cfg,
            output_filename=args.output,
            method=args.method,
            sigma=args.sigma,
            make_png_flag=args.png,
            verbose=args.verbose,
        )
        if args.verbose:
            print(f"[mkmasterbias] Done. Master bias at: {out_path}")
    except Exception as e:
        sys.exit(f"[mkmasterbias] ERROR: {e}")


if __name__ == "__main__":
    main()
