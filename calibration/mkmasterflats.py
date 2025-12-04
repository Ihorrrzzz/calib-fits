# calibration/mkmasterflats.py

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from astropy.io import fits

try:
    from .calib_config import CalibConfig
except ImportError:
    from calibration.calib_config import CalibConfig


def _read_filenames(arg: str) -> List[str]:
    """
    If *arg* is a .lst file, read filenames from it.
    Otherwise treat *arg* as a space-separated list of filenames.
    """
    p = Path(arg)
    if p.suffix == ".lst" and p.is_file():
        with p.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    # space-separated filenames
    return arg.split()


def _get_filter_from_header(hdr, cfg: CalibConfig) -> str:
    filters_keyword = cfg.get("HEADER_SPECIFICATION", "filters_keyword", "FILTER")
    allowed = cfg.get("HEADER_SPECIFICATION", "filters", [])
    val = str(hdr.get(filters_keyword, "")).strip()
    if not allowed:
        return val or "UNKNOWN"
    return val if val in allowed else "UNKNOWN"


def _normalize_flat(data: np.ndarray) -> np.ndarray:
    avg = float(np.mean(data))
    if avg == 0:
        return data
    return data / avg


def create_master_flats(
    files: List[str],
    cfg: CalibConfig,
    verbose: bool = False,
) -> Dict[str, Tuple[str, str]]:
    """
    Create (or load from library) master flats for each filter.

    Returns dict:
        { filter_name: (masterflat_path, normalized_masterflat_path) }
    """
    full_cfg = cfg.config

    flat_enabled = full_cfg.get("IMAGE_PROCESSING", {}).get("flat_correction", True)
    flat_cfg = full_cfg.get("FLAT_CORRECTION", {})
    use_flat_library = bool(flat_cfg.get("library_files", False))

    working_dir = Path(cfg.get("DATA_STRUCTURE", "working_dir", "./work"))
    results_aux_dir = Path(cfg.get("DATA_STRUCTURE", "results_aux_dir", "./results/aux"))
    working_dir.mkdir(parents=True, exist_ok=True)
    results_aux_dir.mkdir(parents=True, exist_ok=True)

    if not flat_enabled:
        if verbose:
            print("[mkmasterflats] Flat correction disabled in config; not creating master flats.")
        return {}

    # --- Library mode -------------------------------------------------
    if use_flat_library:
        results: Dict[str, Tuple[str, str]] = {}

        for base_dir in (results_aux_dir, working_dir):
            if not base_dir.exists():
                continue
            for norm_file in base_dir.glob("masterflat_*_norm.fits"):
                stem = norm_file.stem  # e.g. masterflat_R_norm
                if not (stem.startswith("masterflat_") and stem.endswith("_norm")):
                    continue
                filt = stem[len("masterflat_") : -len("_norm")]
                flat_candidates = [
                    base_dir / f"masterflat_{filt}.fits",
                    working_dir / f"masterflat_{filt}.fits",
                    results_aux_dir / f"masterflat_{filt}.fits",
                ]
                flat_file = None
                for c in flat_candidates:
                    if c.is_file():
                        flat_file = c
                        break
                if flat_file is None:
                    if verbose:
                        print(
                            f"[mkmasterflats] Found normalized library flat {norm_file} "
                            f"but no matching masterflat_{filt}.fits"
                        )
                    continue

                results[filt] = (str(flat_file), str(norm_file))
                if verbose:
                    print(
                        f"[mkmasterflats] Using library flats for filter {filt}: "
                        f"{flat_file}, {norm_file}"
                    )

        if not results and verbose:
            print(
                "[mkmasterflats] FLAT_CORRECTION.library_files=True "
                "but no masterflat_*_norm.fits found in library dirs."
            )
        return results

    # --- Compute from input frames -----------------------------------
    image_type_key = cfg.get("HEADER_SPECIFICATION", "image_type_keyword", "IMAGETYP")
    flat_min_value = float(flat_cfg.get("flat_min_value", 0.0))
    flat_max_value = float(flat_cfg.get("flat_max_value", 0.0))  # 0 => no upper limit

    filter_groups: Dict[str, List[np.ndarray]] = {}
    header_by_filter: Dict[str, object] = {}
    shape_by_filter: Dict[str, tuple] = {}

    for fname in files:
        p = Path(fname)
        if not p.is_file():
            if verbose:
                print(f"[mkmasterflats] Skipping missing file: {fname}")
            continue

        with fits.open(fname, memmap=False) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data.astype(np.float32)

        imagetyp = str(hdr.get(image_type_key, "")).strip().upper()
        if imagetyp != "FLAT":
            continue

        filt = _get_filter_from_header(hdr, cfg)
        if filt == "UNKNOWN":
            if verbose:
                print(f"[mkmasterflats] Skipping {fname}: unknown filter.")
            continue

        mean_val = float(np.mean(data))
        if mean_val < flat_min_value:
            if verbose:
                print(
                    f"[mkmasterflats] Skipping {fname}: mean={mean_val:.1f} < flat_min_value={flat_min_value}."
                )
            continue
        if flat_max_value > 0 and mean_val > flat_max_value:
            if verbose:
                print(
                    f"[mkmasterflats] Skipping {fname}: mean={mean_val:.1f} > flat_max_value={flat_max_value}."
                )
            continue

        if filt not in shape_by_filter:
            shape_by_filter[filt] = data.shape
        elif data.shape != shape_by_filter[filt]:
            if verbose:
                print(
                    f"[mkmasterflats] Skipping {fname}: shape {data.shape} != expected {shape_by_filter[filt]} for {filt}"
                )
            continue

        filter_groups.setdefault(filt, []).append(data)
        header_by_filter[filt] = hdr

    results: Dict[str, Tuple[str, str]] = {}

    for filt, stack_list in filter_groups.items():
        if len(stack_list) < 1:
            if verbose:
                print(f"[mkmasterflats] Filter {filt}: no usable flats.")
            continue

        if verbose:
            print(f"[mkmasterflats] Combining {len(stack_list)} flats for filter {filt}")

        stack = np.stack(stack_list, axis=0)
        median_flat = np.median(stack, axis=0)

        norm_stack = np.stack([_normalize_flat(d) for d in stack_list], axis=0)
        median_norm_flat = np.median(norm_stack, axis=0)

        flat_path = working_dir / f"masterflat_{filt}.fits"
        norm_flat_path = working_dir / f"masterflat_{filt}_norm.fits"

        hdu = fits.PrimaryHDU(median_flat.astype(np.float32), header=header_by_filter[filt])
        hdu.writeto(flat_path, overwrite=True)

        hdun = fits.PrimaryHDU(median_norm_flat.astype(np.float32), header=header_by_filter[filt])
        hdun.writeto(norm_flat_path, overwrite=True)

        # copy to results/aux for possible library use
        aux_flat = results_aux_dir / flat_path.name
        aux_norm = results_aux_dir / norm_flat_path.name
        aux_flat.write_bytes(flat_path.read_bytes())
        aux_norm.write_bytes(norm_flat_path.read_bytes())

        if verbose:
            print(f"[mkmasterflats] Master flat for {filt}: {flat_path}")
            print(f"[mkmasterflats] Normalized master flat for {filt}: {norm_flat_path}")

        results[filt] = (str(flat_path), str(norm_flat_path))

    return results


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create or load master flats from config.")
    parser.add_argument("config", help="Path to config .ini file")
    parser.add_argument(
        "files",
        nargs="+",
        help="Either .lst file with flat list, or one/many flat FITS files",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args(argv)

    cfg = CalibConfig(args.config)

    if len(args.files) == 1:
        files = _read_filenames(args.files[0])
    else:
        files = args.files

    try:
        res = create_master_flats(files, cfg, verbose=args.verbose)
        if args.verbose:
            print(f"[mkmasterflats] Filters processed: {list(res.keys())}")
    except Exception as e:
        sys.exit(f"[mkmasterflats] ERROR: {e}")


if __name__ == "__main__":
    main()
