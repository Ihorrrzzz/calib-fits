# calibration/bias_correction.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.io import fits

try:
    from .calib_config import CalibConfig
except ImportError:
    from calibration.calib_config import CalibConfig


def _read_list(list_path: Path) -> List[str]:
    with list_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_list(list_path: Path, items: List[str]) -> None:
    with list_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it + "\n")


def apply_bias_correction(
    list_in: str,
    list_out: str,
    masterbias_file: str,
    cfg: CalibConfig,
    verbose: bool = False,
) -> None:
    """
    Subtract masterbias from all frames that are NOT bias frames.
    Uses HEADER_SPECIFICATION.image_type_keyword and BIAS_SUBTRACTION.bias_keyword.
    """
    masterbias_path = Path(masterbias_file)
    if not masterbias_path.is_file():
        raise FileNotFoundError(f"Master bias file not found: {masterbias_path}")

    with fits.open(masterbias_path, memmap=False) as hdul:
        mb_data = hdul[0].data.astype(np.float32)

    image_type_key = cfg.get("HEADER_SPECIFICATION", "image_type_keyword", "IMAGETYP")
    bias_label = str(cfg.get("BIAS_SUBTRACTION", "bias_keyword", "BIAS")).strip().upper()

    files_in = _read_list(Path(list_in))
    out_files: List[str] = []

    for fname in files_in:
        p = Path(fname)
        if not p.is_file():
            if verbose:
                print(f"[bias_correction] Missing file: {fname}")
            continue

        try:
            with fits.open(p, memmap=False) as hdul:
                hdr = hdul[0].header
                data = hdul[0].data.astype(np.float32)

            imagetyp = str(hdr.get(image_type_key, "")).strip().upper()
            if imagetyp == bias_label:
                # don't subtract bias from bias frames
                if verbose:
                    print(f"[bias_correction] Skipping bias frame: {fname}")
                continue

            corrected = data - mb_data

            # build output name: stem-b.fits
            new_stem = p.stem
            if not new_stem.endswith("-b"):
                new_stem = new_stem + "-b"
            out_path = p.with_name(new_stem + p.suffix)

            hdu = fits.PrimaryHDU(corrected.astype(np.float32), header=hdr)
            hdu.writeto(out_path, overwrite=True)

            if verbose:
                print(f"[bias_correction] Wrote {out_path}")

            out_files.append(str(out_path.resolve()))
        except Exception as e:
            print(f"[bias_correction] Skipping {fname}: {e}")

    if out_files:
        _write_list(Path(list_out), out_files)
        if verbose:
            print(f"[bias_correction] Output list written to {list_out}")
    else:
        if verbose:
            print("[bias_correction] No files were bias-corrected.")


def main(argv: Optional[list[str]] = None) -> None:
    if len(sys.argv) < 5:
        print(
            "Usage: python bias_correction.py "
            "<list_of_input_files> <list_of_output_files> "
            "<path_to_masterbias_file> <path_to_config_file>"
        )
        sys.exit(1)

    list_in = sys.argv[1]
    list_out = sys.argv[2]
    masterbias_file = sys.argv[3]
    config_file = sys.argv[4]
    verbose = True  # you can change to False if you want it quieter

    cfg = CalibConfig(config_file)
    try:
        apply_bias_correction(list_in, list_out, masterbias_file, cfg, verbose=verbose)
    except Exception as e:
        sys.exit(f"[bias_correction] ERROR: {e}")


if __name__ == "__main__":
    main()
