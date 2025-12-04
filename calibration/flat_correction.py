# calibration/flat_correction.py

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from astropy.io import fits
import shutil

from calib_config import CalibConfig


def _get_image_type(header, cfg: CalibConfig) -> str:
    key = cfg.get("HEADER_SPECIFICATION", "image_type_keyword", "IMAGETYP")
    return str(header.get(key, "")).strip().upper()


def _get_filter(header, cfg: CalibConfig) -> str:
    key = cfg.get("HEADER_SPECIFICATION", "filters_keyword", "FILTER")
    return str(header.get(key, "")).strip()


def _get_object_label(cfg: CalibConfig) -> str:
    image_types = cfg.get("HEADER_SPECIFICATION", "image_types", [])
    obj_label = None
    for val in image_types or []:
        if "OBJECT" in str(val).strip().upper():
            obj_label = str(val).strip().upper()
            break
    if obj_label:
        return obj_label
    return "OBJECT"


def apply_flat_correction(
    input_files: List[str],
    cfg: CalibConfig,
    verbose: bool = False,
) -> List[str]:
    """
    Divide all OBJECT frames by normalized master flats for the
    corresponding filter. Writes '-bdf' style files and also copies
    them into results_dir.

    Returns list of final calibrated OBJECT frame paths in results_dir.
    """
    full_cfg = cfg.config
    flat_enabled = full_cfg.get("IMAGE_PROCESSING", {}).get("flat_correction", True)
    if not flat_enabled:
        if verbose:
            print("[flat_correction] Flat correction disabled in config; nothing to do.")
        return []

    working_dir = Path(cfg.get("DATA_STRUCTURE", "working_dir", "./work"))
    results_dir = Path(cfg.get("DATA_STRUCTURE", "results_dir", "./results"))
    working_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    object_label = _get_object_label(cfg)

    # We will lazily cache masterflat_norm per filter
    masterflat_cache = {}

    final_files: List[str] = []

    for fname in input_files:
        try:
            fpath = Path(fname)
            with fits.open(fpath, memmap=True) as hdul:
                header = hdul[0].header
                imagetyp = _get_image_type(header, cfg)

                if imagetyp != object_label:
                    if verbose:
                        print(f"[flat_correction] Skipping non-object frame {fname} (type={imagetyp}).")
                    continue

                filt = _get_filter(header, cfg)
                if not filt:
                    filt = "UNKNOWN"

                norm_flat_path = working_dir / f"masterflat_{filt}_norm.fits"
                if filt not in masterflat_cache:
                    if not norm_flat_path.is_file():
                        if verbose:
                            print(
                                f"[flat_correction] No normalized masterflat for filter {filt}: "
                                f"{norm_flat_path} – skipping {fname}"
                            )
                        continue
                    with fits.open(norm_flat_path, memmap=True) as hdul_mf:
                        masterflat_norm = hdul_mf[0].data.astype(np.float32)
                    masterflat_cache[filt] = masterflat_norm
                else:
                    masterflat_norm = masterflat_cache[filt]

                data = hdul[0].data.astype(np.float32)

                with np.errstate(divide="ignore", invalid="ignore"):
                    corrected = data / masterflat_norm

                # Name: if '-bd', append 'f' => '-bdf', else just '-f'
                stem = fpath.stem
                if stem.endswith("bd"):
                    new_stem = stem + "f"  # 'file-bd' -> 'file-bdf'
                else:
                    new_stem = stem + "-f"
                new_name = fpath.with_name(new_stem + fpath.suffix)

                hdu = fits.PrimaryHDU(corrected.astype(np.float32), header=header)
                hdr = hdu.header
                hdr["HISTORY"] = "Flat-field correction applied"
                hdr["MF_FILE"] = (norm_flat_path.name, "Normalized master flat")

                # Write next to input file
                hdu.writeto(new_name, overwrite=True)

                # Copy to results_dir as final calibrated science frame
                final_dest = results_dir / new_name.name
                shutil.copy2(new_name, final_dest)

                final_files.append(str(final_dest))

                if verbose:
                    print(f"[flat_correction] Wrote calibrated frame: {final_dest}")

        except Exception as e:
            if verbose:
                print(f"[flat_correction] Error processing {fname}: {e}")

    if verbose:
        print(f"[flat_correction] Total calibrated OBJECT frames: {len(final_files)}")

    return final_files
