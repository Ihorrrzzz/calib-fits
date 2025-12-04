# calibration/bias_correction.py

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


def _get_bias_label(cfg: CalibConfig) -> str:
    val = cfg.get("BIAS_SUBTRACTION", "bias_keyword")
    if val is not None:
        return str(val).strip().upper()
    val = cfg.get("HEADER_SPECIFICATION", "bias_keyword")
    if val is not None:
        return str(val).strip().upper()
    return "BIAS"


def apply_bias_correction(
    input_files: List[str],
    masterbias_path: str,
    cfg: CalibConfig,
    verbose: bool = False,
) -> List[str]:
    """
    Subtract master bias from all non-bias frames.

    Returns list of newly created bias-corrected file paths.
    """
    bias_label = _get_bias_label(cfg)
    masterbias_path = Path(masterbias_path)

    if not masterbias_path.is_file():
        raise FileNotFoundError(f"Master bias not found: {masterbias_path}")

    if verbose:
        print(f"[bias_correction] Using master bias: {masterbias_path}")

    with fits.open(masterbias_path, memmap=True) as hdul_mb:
        mb_data = hdul_mb[0].data.astype(np.float32)

    corrected_files: List[str] = []

    for fname in input_files:
        try:
            fname_path = Path(fname)
            with fits.open(fname_path, memmap=True) as hdul:
                header = hdul[0].header
                imagetyp = _get_image_type(header, cfg)

                if imagetyp == bias_label:
                    # Do not bias-subtract bias frames themselves
                    if verbose:
                        print(f"[bias_correction] Skipping bias frame: {fname}")
                    continue

                data = hdul[0].data.astype(np.float32)
                corrected = data - mb_data

                new_name = fname_path.with_name(fname_path.stem + "-b" + fname_path.suffix)
                hdu = fits.PrimaryHDU(corrected.astype(np.float32), header=header)
                hdr = hdu.header
                hdr["HISTORY"] = "Bias subtraction applied"
                hdr["MB_FILE"] = (masterbias_path.name, "Master bias file")

                hdu.writeto(new_name, overwrite=True)
                corrected_files.append(str(new_name))

                if verbose:
                    print(f"[bias_correction] Wrote bias-corrected frame: {new_name}")

        except Exception as e:
            if verbose:
                print(f"[bias_correction] Error processing {fname}: {e}")

    if verbose:
        print(f"[bias_correction] Created {len(corrected_files)} bias-corrected frames.")
    return corrected_files


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(
        description="Apply master bias to a list of FITS frames (cli version)."
    )
    parser.add_argument(
        "-l",
        "--list",
        required=True,
        help="Text file with FITS paths (one per line).",
    )
    parser.add_argument(
        "-b",
        "--masterbias",
        required=True,
        help="Path to master bias FITS file.",
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

    apply_bias_correction(files, args.masterbias, cfg, verbose=args.verbose)


if __name__ == "__main__":
    _cli()
