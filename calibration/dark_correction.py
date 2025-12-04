# calibration/dark_correction.py

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


def apply_dark_correction(
    list_in: str,
    list_out: str,
    masterdark_file: str,
    cfg: CalibConfig,
    verbose: bool = False,
) -> List[str]:
    """
    Subtract scaled masterdark from all non-bias, non-dark frames.

    Master dark is assumed to be "per second" if created via ScaledExposure* methods,
    thus we do: data - exposure * masterdark.

    Returns list of output filenames that were dark-corrected.
    """
    md_path = Path(masterdark_file)
    if not md_path.is_file():
        raise FileNotFoundError(f"Master dark file not found: {md_path}")

    with fits.open(md_path, memmap=False) as hdul:
        md_data = hdul[0].data.astype(np.float32)

    image_type_key = cfg.get("HEADER_SPECIFICATION", "image_type_keyword", "IMAGETYP")
    exposure_key = cfg.get("HEADER_SPECIFICATION", "exposure_keyword", "EXPTIME")

    bias_label = str(cfg.get("BIAS_SUBTRACTION", "bias_keyword", "BIAS")).strip().upper()
    dark_label = str(cfg.get("DARK_SUBTRACTION", "dark_keyword", "DARK")).strip().upper()

    files_in = _read_list(Path(list_in))
    out_files: List[str] = []

    for fname in files_in:
        p = Path(fname)
        if not p.is_file():
            if verbose:
                print(f"[dark_correction] Missing file: {fname}")
            continue

        try:
            with fits.open(p, memmap=False) as hdul:
                hdr = hdul[0].header
                data = hdul[0].data.astype(np.float32)

            imagetyp = str(hdr.get(image_type_key, "")).strip().upper()
            if imagetyp in (bias_label, dark_label):
                if verbose:
                    print(f"[dark_correction] Skipping {fname} (type={imagetyp})")
                continue

            exptime = float(hdr.get(exposure_key, 0.0))
            if exptime <= 0:
                if verbose:
                    print(f"[dark_correction] Skipping {fname}: invalid exposure {exptime}")
                continue

            corrected = data - exptime * md_data

            stem = p.stem
            # remove -b if present, then append -bd
            if stem.endswith("-b"):
                stem = stem[:-2]
            new_stem = stem + "-bd"
            out_path = p.with_name(new_stem + p.suffix)

            hdu = fits.PrimaryHDU(corrected.astype(np.float32), header=hdr)
            hdu.writeto(out_path, overwrite=True)

            if verbose:
                print(f"[dark_correction] Wrote {out_path}")

            out_files.append(str(out_path.resolve()))

        except Exception as e:
            print(f"[dark_correction] Skipping {fname}: {e}")

    if out_files:
        _write_list(Path(list_out), out_files)
        if verbose:
            print(f"[dark_correction] Output list written to {list_out}")
        return out_files
    else:
        if verbose:
            print("[dark_correction] No files were dark-corrected.")
        return []


def main(argv: Optional[list[str]] = None) -> None:
    if len(sys.argv) < 5:
        print(
            "Usage: python dark_correction.py "
            "<list_of_input_files> <list_of_output_files> "
            "<path_to_masterdark_file> <path_to_config_file>"
        )
        sys.exit(1)

    list_in = sys.argv[1]
    list_out = sys.argv[2]
    masterdark_file = sys.argv[3]
    config_file = sys.argv[4]
    verbose = True  # adjust if needed

    cfg = CalibConfig(config_file)
    try:
        apply_dark_correction(list_in, list_out, masterdark_file, cfg, verbose=verbose)
    except Exception as e:
        sys.exit(f"[dark_correction] ERROR: {e}")


if __name__ == "__main__":
    main()