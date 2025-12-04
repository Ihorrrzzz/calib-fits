# calibration/mkmasterdark.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt

try:
    from .calib_config import CalibConfig
except ImportError:
    from calibration.calib_config import CalibConfig


def find_dark_frames(files: List[str], cfg: CalibConfig, verbose: bool = False) -> List[str]:
    """
    Select dark frames using HEADER_SPECIFICATION.image_type_keyword
    and DARK_SUBTRACTION.dark_keyword.
    """
    image_type_key = cfg.get("HEADER_SPECIFICATION", "image_type_keyword", "IMAGETYP")
    expected_dark = str(cfg.get("DARK_SUBTRACTION", "dark_keyword", "DARK")).strip().upper()

    dark_files: List[str] = []
    for fname in files:
        try:
            with fits.open(fname, memmap=False) as hdul:
                hdr = hdul[0].header
                imagetyp = str(hdr.get(image_type_key, "")).strip().upper()
                if imagetyp == expected_dark:
                    dark_files.append(fname)
        except Exception as e:
            if verbose:
                print(f"[mkmasterdark] Skipping {fname}: {e}")

    if verbose:
        print(f"[mkmasterdark] Found {len(dark_files)} dark frames.")
    return sorted(dark_files)


def make_png(fits_path: Path, png_path: Path, verbose: bool = False) -> None:
    """Create a PNG preview of the master dark."""
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    plt.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
    plt.colorbar(label="Pixel value (zscale)")
    plt.title("Master Dark")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[mkmasterdark] PNG saved to {png_path}")


def make_master_dark(
    all_files: List[str],
    cfg: CalibConfig,
    output_filename: str = "masterdark.fits",
    method: Optional[str] = None,
    make_png_flag: bool = False,
    verbose: bool = False,
) -> str:
    """
    Create a master dark frame from all_files, OR load it from library
    depending on the config.

    Returns full path to master dark FITS file.
    """
    full_cfg = cfg.config
    working_dir = Path(cfg.get("DATA_STRUCTURE", "working_dir", "./work"))
    results_aux_dir = Path(cfg.get("DATA_STRUCTURE", "results_aux_dir", "./results/aux"))
    working_dir.mkdir(parents=True, exist_ok=True)
    results_aux_dir.mkdir(parents=True, exist_ok=True)

    dark_enabled = full_cfg.get("IMAGE_PROCESSING", {}).get("dark_correction", True)

    # --- Library mode for darks --------------------------------------
    dark_lib_cfg = full_cfg.get("DARK_SUBTRACTION", {})
    use_dark_library = bool(dark_lib_cfg.get("library_files", False))

    if use_dark_library:
        flat_cfg = full_cfg.get("FLAT_CORRECTION", {})
        lib_dark_name = flat_cfg.get("library_dark", "masterdark.fits")

        candidates = [
            results_aux_dir / lib_dark_name,
            working_dir / lib_dark_name,
        ]
        for c in candidates:
            if c.is_file():
                if verbose:
                    print(f"[mkmasterdark] Using library master dark: {c}")
                return str(c)

        raise FileNotFoundError(
            f"[mkmasterdark] DARK_SUBTRACTION.library_files=True but "
            f"no library dark '{lib_dark_name}' found in {results_aux_dir} or {working_dir}"
        )

    # --- Normal mode: build from dark frames -------------------------
    if not dark_enabled:
        if verbose:
            print("[mkmasterdark] Dark correction disabled in config; not creating master dark.")
        return str(working_dir / output_filename)

    method_cfg = (
        method
        or full_cfg.get("IMAGE_PROCESSING", {}).get("dark_correction_method")
        or full_cfg.get("DARK_SUBTRACTION", {}).get("option")
        or "ScaledExposureMedian"
    )

    dark_files = find_dark_frames(all_files, cfg, verbose=verbose)
    if not dark_files:
        raise RuntimeError("[mkmasterdark] No dark frames found to build master dark.")

    exptime_key = cfg.get("HEADER_SPECIFICATION", "exposure_keyword", "EXPTIME")

    data_list = []
    exptime_list = []

    for fname in dark_files:
        with fits.open(fname, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float32)
            exp = float(hdul[0].header.get(exptime_key, 0.0))
            if exp <= 0:
                if verbose:
                    print(f"[mkmasterdark] Skipping {fname}: invalid exposure {exp}")
                continue
            data_list.append(data)
            exptime_list.append(exp)

    if not data_list:
        raise RuntimeError("[mkmasterdark] No valid dark frames left after exposure checks.")

    stack = np.stack(data_list, axis=0)
    exptimes = np.array(exptime_list, dtype=np.float32)

    if method_cfg == "ScaledExposureMedian":
        scaled = stack / exptimes[:, None, None]
        master_dark = np.median(scaled, axis=0)
    elif method_cfg == "ScaledExposureAverage":
        scaled = stack / exptimes[:, None, None]
        master_dark = np.mean(scaled, axis=0)
    elif method_cfg == "EqualExposure":
        # assume same exposure, just median
        master_dark = np.median(stack, axis=0)
    else:
        if verbose:
            print(f"[mkmasterdark] Unsupported method '{method_cfg}', falling back to ScaledExposureMedian.")
        scaled = stack / exptimes[:, None, None]
        master_dark = np.median(scaled, axis=0)

    out_path = working_dir / output_filename
    hdu = fits.PrimaryHDU(master_dark.astype(np.float32))
    hdu.header["MD_COMB"] = (method_cfg, "Master dark combination method")
    hdu.writeto(out_path, overwrite=True)

    aux_path = results_aux_dir / output_filename
    aux_path.write_bytes(out_path.read_bytes())

    if verbose:
        print(f"[mkmasterdark] Master dark written to {out_path}")
        print(f"[mkmasterdark] Copy stored to {aux_path}")

    if make_png_flag:
        png_path = out_path.with_suffix(".png")
        make_png(out_path, png_path, verbose=verbose)

    return str(out_path)


def _read_list_file(list_path: Path) -> List[str]:
    with list_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create a master dark frame using settings from config.ini"
    )
    parser.add_argument("-l", "--list", required=True, help="Text file with list of FITS files")
    parser.add_argument("-o", "--output", required=True, help="Output master dark filename")
    parser.add_argument("-m", "--method", help="Override dark correction method")
    parser.add_argument("-c", "--config", default="config.ini", help="Path to config .ini file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-p", "--png", action="store_true", help="Create PNG preview")
    args = parser.parse_args(argv)

    cfg = CalibConfig(args.config)
    files = _read_list_file(Path(args.list))

    try:
        out_path = make_master_dark(
            all_files=files,
            cfg=cfg,
            output_filename=args.output,
            method=args.method,
            make_png_flag=args.png,
            verbose=args.verbose,
        )
        if args.verbose:
            print(f"[mkmasterdark] Done. Master dark at: {out_path}")
    except Exception as e:
        sys.exit(f"[mkmasterdark] ERROR: {e}")


if __name__ == "__main__":
    main()
