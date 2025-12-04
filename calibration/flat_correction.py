# calibration/flat_correction.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

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


def apply_flat_correction(
    list_in: str,
    list_out: str,
    cfg: CalibConfig,
    verbose: bool = False,
) -> None:
    """
    Divide science (OBJECT) images by normalized master flats.

    Uses:
      - HEADER_SPECIFICATION.image_type_keyword
      - HEADER_SPECIFICATION.filters_keyword
      - HEADER_SPECIFICATION.filters (valid filters)
      - DATA_STRUCTURE.working_dir / results_dir / results_aux_dir
    """
    working_dir = Path(cfg.get("DATA_STRUCTURE", "working_dir", "./work"))
    results_dir = Path(cfg.get("DATA_STRUCTURE", "results_dir", "./results"))
    results_aux_dir = Path(cfg.get("DATA_STRUCTURE", "results_aux_dir", "./results/aux"))
    working_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_aux_dir.mkdir(parents=True, exist_ok=True)

    full_cfg = cfg.config
    image_type_key = cfg.get("HEADER_SPECIFICATION", "image_type_keyword", "IMAGETYP")
    filter_key = cfg.get("HEADER_SPECIFICATION", "filters_keyword", "FILTER")
    valid_filters = set(cfg.get("HEADER_SPECIFICATION", "filters", []))

    list_path = Path(list_in)
    input_files = _read_list(list_path)

    masterflat_cache: Dict[str, np.ndarray] = {}
    output_files: List[str] = []

    for fname in input_files:
        p = Path(fname)
        if not p.is_file():
            if verbose:
                print(f"[flat_correction] Missing file: {fname}")
            continue

        try:
            with fits.open(p, memmap=False) as hdul:
                hdr = hdul[0].header
                data = hdul[0].data.astype(np.float32)

            imagetyp = str(hdr.get(image_type_key, "")).strip().upper()
            if imagetyp != "OBJECT":
                # Only calibrate science frames here
                continue

            filt = str(hdr.get(filter_key, "")).strip()
            if valid_filters and filt not in valid_filters:
                if verbose:
                    print(f"[flat_correction] Skipping {fname}: invalid filter '{filt}'")
                continue

            # Find normalized masterflat for this filter
            if filt not in masterflat_cache:
                norm_name = f"masterflat_{filt}_norm.fits"
                norm_flat_path = working_dir / norm_name
                if not norm_flat_path.is_file():
                    alt = results_aux_dir / norm_name
                    if alt.is_file():
                        norm_flat_path = alt
                    else:
                        if verbose:
                            print(
                                f"[flat_correction] No normalized masterflat for filter {filt}: "
                                f"{norm_flat_path} or {alt} – skipping {fname}"
                            )
                        continue

                if verbose:
                    print(f"[flat_correction] Loading normalized masterflat for {filt}: {norm_flat_path}")
                with fits.open(norm_flat_path, memmap=False) as fh:
                    masterflat_cache[filt] = fh[0].data.astype(np.float32)

            norm_flat = masterflat_cache[filt]
            # avoid division by zero:
            with np.errstate(divide="ignore", invalid="ignore"):
                calibrated = data / norm_flat
                calibrated[~np.isfinite(calibrated)] = 0.0

            stem = p.stem
            # remove -bd if present, then append -bdf
            if stem.endswith("-bd"):
                stem = stem[:-3]
            out_name = stem + "-bdf" + p.suffix
            out_path = results_dir / out_name

            hdu = fits.PrimaryHDU(calibrated.astype(np.float32), header=hdr)
            hdu.writeto(out_path, overwrite=True)
            if verbose:
                print(f"[flat_correction] Wrote {out_path}")

            output_files.append(str(out_path.resolve()))

        except Exception as e:
            print(f"[flat_correction] Skipping {fname}: {e}")

    if output_files:
        _write_list(Path(list_out), output_files)
        if verbose:
            print(f"[flat_correction] Output list written to {list_out}")
            print("[flat_correction] Calibrated science images are in results_dir.")
    else:
        if verbose:
            print("[flat_correction] No files were flat-corrected.")


def main(argv: Optional[list[str]] = None) -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: python flat_correction.py "
            "<list_of_input_files> <list_of_output_files> <path_to_config_file>"
        )
        sys.exit(1)

    list_in = sys.argv[1]
    list_out = sys.argv[2]
    config_file = sys.argv[3]
    verbose = True  # adjust if needed

    cfg = CalibConfig(config_file)
    try:
        apply_flat_correction(list_in, list_out, cfg, verbose=verbose)
    except Exception as e:
        sys.exit(f"[flat_correction] ERROR: {e}")


if __name__ == "__main__":
    main()
