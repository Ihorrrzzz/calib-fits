# calibration/mkmasterflats.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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


def _get_flat_label(cfg: CalibConfig) -> str:
    image_types = cfg.get("HEADER_SPECIFICATION", "image_types", [])
    # Try to find something containing 'FLAT'
    flat_label = None
    for val in image_types or []:
        if "FLAT" in str(val).strip().upper():
            flat_label = str(val).strip().upper()
            break
    if flat_label:
        return flat_label
    return "FLAT"


def create_master_flats(
    input_files: List[str],
    cfg: CalibConfig,
    verbose: bool = False,
) -> Dict[str, Tuple[str, str]]:
    """
    Create master flats and normalized master flats per filter.

    Returns:
      dict[filter] = (masterflat_path, masterflat_norm_path)
    """
    full_cfg = cfg.config
    flat_enabled = full_cfg.get("IMAGE_PROCESSING", {}).get("flat_correction", True)
    if not flat_enabled:
        if verbose:
            print("[mkmasterflats] Flat correction disabled in config; not creating master flats.")
        return {}

    working_dir = Path(cfg.get("DATA_STRUCTURE", "working_dir", "./work"))
    results_aux_dir = Path(cfg.get("DATA_STRUCTURE", "results_aux_dir", "./results/aux"))
    working_dir.mkdir(parents=True, exist_ok=True)
    results_aux_dir.mkdir(parents=True, exist_ok=True)

    flat_label = _get_flat_label(cfg)
    flat_min = full_cfg.get("FLAT_CORRECTION", {}).get("flat_min_value", 0)
    flat_max = full_cfg.get("FLAT_CORRECTION", {}).get("flat_max_value", 0)
    check_consistency = full_cfg.get("FLAT_CORRECTION", {}).get("check_flat_consistency", False)

    frames_by_filter: Dict[str, List[np.ndarray]] = {}
    shapes_by_filter: Dict[str, Tuple[int, int]] = {}

    for fname in input_files:
        try:
            with fits.open(fname, memmap=True) as hdul:
                header = hdul[0].header
                imagetyp = _get_image_type(header, cfg)
                if imagetyp != flat_label:
                    continue

                filt = _get_filter(header, cfg)
                if not filt:
                    filt = "UNKNOWN"

                data = hdul[0].data.astype(np.float32)

                dmin = float(np.min(data))
                dmax = float(np.max(data))

                if dmin < flat_min:
                    if verbose:
                        print(
                            f"[mkmasterflats] Rejecting flat {fname} (filter={filt}): "
                            f"min={dmin} < flat_min={flat_min}"
                        )
                    continue
                if flat_max > 0 and dmax > flat_max:
                    if verbose:
                        print(
                            f"[mkmasterflats] Rejecting flat {fname} (filter={filt}): "
                            f"max={dmax} > flat_max={flat_max}"
                        )
                    continue

                if check_consistency:
                    shape = data.shape
                    if filt not in shapes_by_filter:
                        shapes_by_filter[filt] = shape
                    elif shapes_by_filter[filt] != shape:
                        if verbose:
                            print(
                                f"[mkmasterflats] Rejecting {fname}: shape {shape} "
                                f"!= expected {shapes_by_filter[filt]} for filter={filt}"
                            )
                        continue

                frames_by_filter.setdefault(filt, []).append(data)

        except Exception as e:
            if verbose:
                print(f"[mkmasterflats] Error reading {fname}: {e}")

    if verbose:
        for filt, frames in frames_by_filter.items():
            print(f"[mkmasterflats] Filter {filt}: {len(frames)} flat frames selected.")

    results: Dict[str, Tuple[str, str]] = {}

    for filt, frames in frames_by_filter.items():
        if len(frames) < 2:
            if verbose:
                print(f"[mkmasterflats] Filter {filt}: not enough flats ({len(frames)}) – skipping.")
            continue

        stack = np.stack(frames, axis=0)
        master_flat = np.median(stack, axis=0)

        # Normalized flats: each frame / its mean, then median
        norm_frames = []
        for frame in frames:
            mean_val = float(np.mean(frame))
            if mean_val <= 0:
                norm_frames.append(frame)
            else:
                norm_frames.append(frame / mean_val)
        norm_stack = np.stack(norm_frames, axis=0)
        master_flat_norm = np.median(norm_stack, axis=0)

        flat_path = working_dir / f"masterflat_{filt}.fits"
        norm_path = working_dir / f"masterflat_{filt}_norm.fits"

        hdu_flat = fits.PrimaryHDU(master_flat.astype(np.float32))
        hdu_flat.header["HISTORY"] = f"Master flat for filter {filt}"
        hdu_flat.header["MF_FILT"] = (filt, "Filter name")

        hdu_norm = fits.PrimaryHDU(master_flat_norm.astype(np.float32))
        hdu_norm.header["HISTORY"] = f"Normalized master flat for filter {filt}"
        hdu_norm.header["MF_FILT"] = (filt, "Filter name")

        hdu_flat.writeto(flat_path, overwrite=True)
        hdu_norm.writeto(norm_path, overwrite=True)

        # Copy into results/aux as permanent calibration files
        flat_copy = results_aux_dir / flat_path.name
        norm_copy = results_aux_dir / norm_path.name
        shutil.copy2(flat_path, flat_copy)
        shutil.copy2(norm_path, norm_copy)

        results[filt] = (str(flat_path), str(norm_path))

        if verbose:
            print(
                f"[mkmasterflats] Filter {filt}: "
                f"masterflat={flat_path}, masterflat_norm={norm_path}"
            )

    return results
