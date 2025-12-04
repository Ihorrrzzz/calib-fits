# calibration/dark_correction.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from astropy.io import fits

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


def _get_bias_label(cfg: CalibConfig) -> str:
    val = cfg.get("BIAS_SUBTRACTION", "bias_keyword")
    if val is not None:
        return str(val).strip().upper()
    return "BIAS"


def _get_dark_label(cfg: CalibConfig) -> str:
    val = cfg.get("DARK_SUBTRACTION", "dark_keyword")
    if val is not None:
        return str(val).strip().upper()
    return "DARK"


def apply_dark_correction(
    input_files: List[str],
    masterdark_path: str,
    cfg: CalibConfig,
    verbose: bool = False,
) -> List[str]:
    """
    Subtract master dark from all non-bias, non-dark frames
    (typically FLAT and OBJECT).
    Returns list of newly created dark-corrected file paths.
    """
    masterdark_path = Path(masterdark_path)
    if not masterdark_path.is_file():
        raise FileNotFoundError(f"Master dark not found: {masterdark_path}")

    full_cfg = cfg.config
    method_cfg = full_cfg.get("IMAGE_PROCESSING", {}).get("dark_correction_method", "ScaledExposureMedian")
    method_lower = str(method_cfg).lower()

    if verbose:
        print(f"[dark_correction] Using master dark: {masterdark_path}")
        print(f"[dark_correction] Method: {method_cfg}")

    with fits.open(masterdark_path, memmap=True) as hdul_md:
        md_data = hdul_md[0].data.astype(np.float32)
        md_hdr = hdul_md[0].header
        per_second = bool(md_hdr.get("MD_PERSEC", False))

    bias_label = _get_bias_label(cfg)
    dark_label = _get_dark_label(cfg)

    corrected_files: List[str] = []

    for fname in input_files:
        try:
            fpath = Path(fname)
            with fits.open(fpath, memmap=True) as hdul:
                header = hdul[0].header
                imagetyp = _get_image_type(header, cfg)

                if imagetyp in {bias_label, dark_label}:
                    if verbose:
                        print(f"[dark_correction] Skipping frame {fname} (type={imagetyp}).")
                    continue

                data = hdul[0].data.astype(np.float32)
                exptime = _get_exptime(header, cfg)
                if exptime <= 0:
                    exptime = 1.0  # safe fallback

                if per_second or method_lower in {
                    "scaledexposuremedian",
                    "scaledexposureaverage",
                }:
                    corrected = data - exptime * md_data
                else:
                    corrected = data - md_data

                # Create name: if already "-b", append "d"; otherwise add "-d"
                stem = fpath.stem
                if stem.endswith("-b"):
                    new_stem = stem + "d"  # 'file-b' -> 'file-bd'
                else:
                    new_stem = stem + "-d"
                new_name = fpath.with_name(new_stem + fpath.suffix)

                hdu = fits.PrimaryHDU(corrected.astype(np.float32), header=header)
                hdr = hdu.header
                hdr["HISTORY"] = "Dark subtraction applied"
                hdr["MD_FILE"] = (masterdark_path.name, "Master dark file")
                hdr["MD_METH"] = (str(method_cfg), "Dark combination method")

                hdu.writeto(new_name, overwrite=True)
                corrected_files.append(str(new_name))

                if verbose:
                    print(f"[dark_correction] Wrote dark-corrected frame: {new_name}")

        except Exception as e:
            if verbose:
                print(f"[dark_correction] Error processing {fname}: {e}")

    if verbose:
        print(f"[dark_correction] Created {len(corrected_files)} dark-corrected frames.")
    return corrected_files


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(
        description="Apply master dark to a list of FITS frames (cli version)."
    )
    parser.add_argument(
        "-l",
        "--list",
        required=True,
        help="Text file with FITS paths (one per line).",
    )
    parser.add_argument(
        "-d",
        "--masterdark",
        required=True,
        help="Path to master dark FITS file.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to LISNYKY_Moravian-C4-16000.ini",
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

    apply_dark_correction(files, args.masterdark, cfg, verbose=args.verbose)


if __name__ == "__main__":
    _cli()
