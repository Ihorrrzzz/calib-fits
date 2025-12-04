# calibration/mkmasterdark.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from calib_config import CalibConfig


def _get_image_type(header, cfg: CalibConfig) -> str:
    key = cfg.get("HEADER_SPECIFICATION", "image_type_keyword", "IMAGETYP")
    return str(header.get(key, "")).strip().upper()


def _get_exptime(header, cfg: CalibConfig) -> float:
    key = cfg.get("HEADER_SPECIFICATION", "exposure_keyword", "EXPTIME")
    try:
        return float(header.get(key, 0.0))
    except Exception:
        return 0.0


def _get_dark_label(cfg: CalibConfig) -> str:
    val = cfg.get("DARK_SUBTRACTION", "dark_keyword")
    if val is not None:
        return str(val).strip().upper()
    val = cfg.get("HEADER_SPECIFICATION", "dark_keyword")
    if val is not None:
        return str(val).strip().upper()
    return "DARK"


def find_dark_frames(
    files: List[str],
    cfg: CalibConfig,
    verbose: bool = False,
) -> List[str]:
    dark_label = _get_dark_label(cfg)
    result: List[str] = []

    for fname in files:
        try:
            with fits.open(fname, memmap=True) as hdul:
                header = hdul[0].header
                imagetyp = _get_image_type(header, cfg)
                if imagetyp == dark_label:
                    result.append(fname)
        except Exception as e:
            if verbose:
                print(f"[mkmasterdark] Skipping {fname}: {e}")

    if verbose:
        print(f"[mkmasterdark] Found {len(result)} dark frames (label = {dark_label}).")
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
    plt.title("Master Dark")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    if verbose:
        print(f"[mkmasterdark] PNG preview saved to {png_path}")

    return str(png_path)


def make_master_dark(
    all_files: List[str],
    cfg: CalibConfig,
    output_filename: str = "masterdark.fits",
    method: Optional[str] = None,
    make_png: bool = False,
    verbose: bool = False,
) -> str:
    """
    Create a master dark frame from all_files, using config-driven settings.
    Returns full path to master dark FITS file.
    """
    full_cfg = cfg.config
    dark_enabled = full_cfg.get("IMAGE_PROCESSING", {}).get("dark_correction", True)
    if not dark_enabled:
        if verbose:
            print("[mkmasterdark] Dark correction disabled in config; not creating master dark.")
        working_dir = Path(cfg.get("DATA_STRUCTURE", "working_dir", "./work"))
        working_dir.mkdir(parents=True, exist_ok=True)
        return str(working_dir / output_filename)

    method_cfg = method or full_cfg.get("IMAGE_PROCESSING", {}).get(
        "dark_correction_method", "ScaledExposureMedian"
    )

    working_dir = Path(cfg.get("DATA_STRUCTURE", "working_dir", "./work"))
    working_dir.mkdir(parents=True, exist_ok=True)
    out_path = working_dir / output_filename

    dark_files = find_dark_frames(all_files, cfg, verbose=verbose)
    if not dark_files:
        raise RuntimeError("[mkmasterdark] No dark frames found; cannot create master dark.")

    if verbose:
        print(f"[mkmasterdark] Combining {len(dark_files)} dark frames using {method_cfg}.")

    data_list = []
    exp_list = []

    for fname in dark_files:
        with fits.open(fname, memmap=True) as hdul:
            header = hdul[0].header
            exptime = _get_exptime(header, cfg)
            data = hdul[0].data.astype(np.float32)
            data_list.append(data)
            exp_list.append(exptime if exptime > 0 else 1.0)

    method_lower = str(method_cfg).lower()
    per_second = False

    if method_lower in {"scaledexposuremedian", "scaledexposureaverage"}:
        # Build per-second dark frames
        scaled = [arr / expt for arr, expt in zip(data_list, exp_list)]
        stack = np.stack(scaled, axis=0)
        if method_lower == "scaledexposureaverage":
            master = np.mean(stack, axis=0)
        else:
            master = np.median(stack, axis=0)
        per_second = True
    else:
        # Simple median combination, assuming exposures are equal / similar
        stack = np.stack(data_list, axis=0)
        master = np.median(stack, axis=0)
        per_second = False
        if verbose:
            print(
                "[mkmasterdark] Using simple median combination without exposure scaling. "
                "Ensure dark exposures are consistent."
            )

    hdu = fits.PrimaryHDU(master.astype(np.float32))
    hdr = hdu.header
    hdr["HISTORY"] = "Master dark created by mkmasterdark.py"
    hdr["MD_NFRM"] = (len(dark_files), "Number of dark frames")
    hdr["MD_METH"] = (str(method_cfg), "Dark combination method")
    hdr["MD_PERSEC"] = (per_second, "True if dark scaled to 1 sec")

    hdu.writeto(out_path, overwrite=True)

    if verbose:
        print(f"[mkmasterdark] Master dark saved to {out_path}")

    if make_png:
        make_png(out_path, verbose=verbose)

    return str(out_path)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(description="Create master dark from list of FITS files.")
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
        default="masterdark.fits",
        help="Output master dark filename (inside working_dir).",
    )
    parser.add_argument(
        "-m",
        "--method",
        help="Override dark combination method (config: IMAGE_PROCESSING.dark_correction_method).",
    )
    parser.add_argument(
        "-p",
        "--png",
        action="store_true",
        help="Create PNG preview of the master dark.",
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

    make_master_dark(
        files,
        cfg,
        output_filename=args.output,
        method=args.method,
        make_png=args.png,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    _cli()
