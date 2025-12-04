from __future__ import annotations

import argparse
import tarfile
import zipfile
import shutil
from pathlib import Path
from typing import Callable, Optional, List

import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, Box2DKernel

from calibration.calib_config import CalibConfig
from calibration import (
    mkmasterbias,
    mkmasterdark,
    mkmasterflats,
    bias_correction,
    dark_correction,
    flat_correction,
)


class CalibrationPipeline:
    """
    High-level orchestrator for the full CCD calibration pipeline:

      RAW (+optional overscan) → master bias → bias-sub
                               → master dark → dark-sub
                               → master flats → flat-sub (science)
                               → optional cosmetic / background cleanup
    """

    FITS_EXTENSIONS = {".fits", ".fit", ".FITS", ".FIT"}

    def __init__(self, config_path: str, root_dir: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        config_path : str
            Path to the .ini configuration file.
        root_dir : str, optional
            If given, all DATA_STRUCTURE dirs (work/results/aux) are created
            INSIDE this directory, ignoring relative paths from the config.
            This is what the GUI uses: the selected raw directory.
        """
        self.config_path = str(config_path)
        self.cfg = CalibConfig(self.config_path)
        self.full_config = self.cfg.config

        self.root_dir: Optional[Path]
        if root_dir is not None:
            self.root_dir = Path(root_dir).expanduser().resolve()
        else:
            self.root_dir = None

        self._init_dirs()

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    def _init_dirs(self) -> None:
        """
        Initialise working / results / aux dirs.

        - If root_dir is provided: use <root_dir>/work, <root_dir>/results, <root_dir>/results/aux.
        - Otherwise: use paths from DATA_STRUCTURE section, interpreted
          relative to the config file location.
        """
        if self.root_dir is not None:
            base = self.root_dir
            self.working_dir = base / "work"
            self.results_dir = base / "results"
            self.results_aux_dir = self.results_dir / "aux"
        else:
            base = Path(self.config_path).expanduser().resolve().parent
            wd = self.cfg.get("DATA_STRUCTURE", "working_dir", "./work")
            rd = self.cfg.get("DATA_STRUCTURE", "results_dir", "./results")
            ra = self.cfg.get("DATA_STRUCTURE", "results_aux_dir", "./results/aux")

            self.working_dir = (base / wd).expanduser().resolve()
            self.results_dir = (base / rd).expanduser().resolve()
            self.results_aux_dir = (base / ra).expanduser().resolve()

        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_aux_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_full_calibration(
        self,
        raw_source: str,
        source_type: str = "directory",  # "directory" | "list" | "archive"
        extra_calib_files: Optional[List[str]] = None,
        verbose: bool = True,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        """
        Run the full calibration pipeline.

        Parameters
        ----------
        raw_source : str
            If source_type == "directory": path to directory with FITS
            If source_type == "list": path to txt/lst file with FITS paths
            If source_type == "archive": .zip/.tar(.gz) with FITS

        source_type : {"directory","list","archive"}
            Type of raw_source.

        extra_calib_files : list of str, optional
            Additional calibration frames (e.g. long darks, extra flats) that
            should be *included when building master* bias/dark/flats, but
            are not part of the main science directory. These files are NOT
            written to the work/results dirs by this function; they are only
            read when creating masters.

        verbose : bool
            Verbose logging for sub-routines.

        log_callback : callable(str) -> None
            If given, used instead of print for log messages.

        Returns
        -------
        list of str
            Paths to final calibrated OBJECT frames (after all steps).
        """
        log = log_callback or print

        # ------------------ Collect inputs -----------------------------
        input_files = self._collect_input_files(raw_source, source_type, log)
        if not input_files:
            raise RuntimeError("No FITS files found for calibration.")

        # Overscan step works on raw files from the directory/list/archive.
        # extra_calib_files are assumed to be already in a good state.
        files_for_pipeline = self._apply_overscan_if_enabled(input_files, log, verbose)

        # Save the base list that flows through the pipeline
        list_raw = self.working_dir / "pipeline_raw.lst"
        self._write_list(list_raw, files_for_pipeline)

        # Combined list only for master creation (includes extra long darks etc.)
        calib_master_files = list(files_for_pipeline)
        if extra_calib_files:
            extra_resolved = [
                str(Path(f).expanduser().resolve()) for f in extra_calib_files
            ]
            calib_master_files = sorted(set(calib_master_files + extra_resolved))
            log(
                f"Including {len(extra_resolved)} extra calibration file(s) "
                "when building master bias/dark/flat."
            )

        img_proc_cfg = self.full_config.get("IMAGE_PROCESSING", {})

        # ------------------ Bias --------------------------------------
        bias_enabled = self._as_bool(img_proc_cfg.get("bias_subtraction", True), True)

        if bias_enabled:
            log("=== CREATING MASTER BIAS ===")
            masterbias_path = mkmasterbias.create_master_bias(
                calib_master_files,
                self.cfg,
                output_filename="masterbias.fits",
                method=img_proc_cfg.get("bias_subtraction_method"),
                sigma=img_proc_cfg.get("bias_subtraction_sigma", 2.5),
                make_png_flag=False,
                verbose=verbose,
            )

            log("=== BIAS CORRECTION ===")
            list_b = self.working_dir / "pipeline_bias_corrected.lst"
            bias_corrected_files = bias_correction.apply_bias_correction(
                list_in=str(list_raw),
                list_out=str(list_b),
                masterbias_file=masterbias_path,
                cfg=self.cfg,
                verbose=verbose,
            )
        else:
            log("Bias subtraction disabled in config; skipping bias steps.")
            bias_corrected_files = list(files_for_pipeline)
            list_b = list_raw

        # ------------------ Dark --------------------------------------
        dark_enabled = self._as_bool(img_proc_cfg.get("dark_correction", True), True)

        if dark_enabled:
            log("=== CREATING MASTER DARK ===")
            masterdark_path = mkmasterdark.make_master_dark(
                calib_master_files,
                self.cfg,
                output_filename="masterdark.fits",
                method=img_proc_cfg.get("dark_correction_method"),
                make_png_flag=False,
                verbose=verbose,
            )

            log("=== DARK CORRECTION ===")
            list_bd = self.working_dir / "pipeline_dark_corrected.lst"
            dark_corrected_files = dark_correction.apply_dark_correction(
                list_in=str(list_b),
                list_out=str(list_bd),
                masterdark_file=masterdark_path,
                cfg=self.cfg,
                verbose=verbose,
            )
        else:
            log("Dark correction disabled in config; skipping dark steps.")
            dark_corrected_files = list(bias_corrected_files)
            list_bd = list_b

        # ------------------ Flats & science ---------------------------
        flat_enabled = self._as_bool(img_proc_cfg.get("flat_correction", True), True)

        if flat_enabled:
            log("=== CREATING MASTER FLATS ===")
            masterflats = mkmasterflats.create_master_flats(
                calib_master_files,
                self.cfg,
                verbose=verbose,
            )

            if verbose:
                for filt, (mf, mfn) in masterflats.items():
                    log(f"Filter {filt}: masterflat={mf}, masterflat_norm={mfn}")

            log("=== FLAT CORRECTION (SCIENCE FRAMES) ===")
            list_bdf = self.working_dir / "pipeline_flat_corrected.lst"
            final_files = flat_correction.apply_flat_correction(
                list_in=str(list_bd),
                list_out=str(list_bdf),
                cfg=self.cfg,
                verbose=verbose,
            )
        else:
            log("Flat correction disabled in config; skipping flats.")
            final_files = list(dark_corrected_files)

        # ------------------ Cosmetic corrections (optional) -----------
        final_files = self._apply_cosmetic_if_enabled(final_files, log, verbose)

        # ------------------ Background modelling (optional) -----------
        final_files = self._apply_background_if_enabled(final_files, log, verbose)

        log("=== CALIBRATION COMPLETE ===")
        log(f"Final calibrated science images are in: {self.results_dir}")
        return final_files

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _write_list(path: Path, items: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(str(it) + "\n")

    @staticmethod
    def _as_bool(value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            v = value.strip().lower()
            return v in {"1", "true", "yes", "y", "on"}
        try:
            return bool(value)
        except Exception:
            return default

    @staticmethod
    def _add_suffix(path: str | Path, suffix: str) -> str:
        p = Path(path)
        return str(p.with_name(p.stem + suffix + p.suffix))

    # ------------------------------------------------------------------
    # Input collection
    # ------------------------------------------------------------------
    def _collect_input_files(
        self,
        raw_source: str,
        source_type: str,
        log: Callable[[str], None],
    ) -> List[str]:
        """
        Resolve raw_source into a list of FITS file paths.
        """
        source_type = source_type.lower()
        p = Path(raw_source).expanduser().resolve()

        if source_type == "directory":
            if not p.is_dir():
                raise FileNotFoundError(f"Directory not found: {p}")
            files = sorted(
                str(f.resolve())
                for f in p.iterdir()
                if f.suffix in self.FITS_EXTENSIONS and f.is_file()
            )
            log(f"Found {len(files)} FITS files in directory {p}")
            return files

        if source_type == "list":
            if not p.is_file():
                raise FileNotFoundError(f"List file not found: {p}")
            with p.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            files: List[str] = []
            for line in lines:
                fp = Path(line).expanduser().resolve()
                if fp.suffix in self.FITS_EXTENSIONS and fp.is_file():
                    files.append(str(fp))
            log(f"Loaded {len(files)} FITS files from list {p}")
            return sorted(files)

        if source_type == "archive":
            if not p.is_file():
                raise FileNotFoundError(f"Archive not found: {p}")
            extract_root = self.working_dir / "archives"
            extract_root.mkdir(parents=True, exist_ok=True)
            target_dir = extract_root / p.stem
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p, "r") as zf:
                    zf.extractall(target_dir)
            elif tarfile.is_tarfile(p):
                with tarfile.open(p, "r:*") as tf:
                    tf.extractall(target_dir)
            else:
                raise ValueError(f"Unsupported archive format: {p}")

            files = [
                str(f.resolve())
                for f in target_dir.rglob("*")
                if f.suffix in self.FITS_EXTENSIONS and f.is_file()
            ]
            files.sort()
            log(f"Extracted archive to {target_dir}, found {len(files)} FITS files.")
            return files

        raise ValueError(f"Unknown source_type: {source_type!r}")

    # ------------------------------------------------------------------
    # Overscan
    # ------------------------------------------------------------------
    def _apply_overscan_if_enabled(
        self,
        input_files: List[str],
        log: Callable[[str], None],
        verbose: bool,
    ) -> List[str]:
        ov_cfg = self.full_config.get("OVERSCAN", {})
        enabled = self._as_bool(ov_cfg.get("enabled", False), False)
        if not enabled:
            return input_files

        region = ov_cfg.get("region")
        if not region:
            log("OVERSCAN.enabled=True but OVERSCAN.region is not set – skipping overscan.")
            return input_files

        try:
            xpart, ypart = region.split(",")
            x1, x2 = map(int, xpart.split(":"))
            y1, y2 = map(int, ypart.split(":"))
        except Exception:
            log(f"OVERSCAN.region='{region}' is invalid – expected 'x1:x2,y1:y2'. Skipping overscan.")
            return input_files

        mode = str(ov_cfg.get("mode", "row")).lower()
        if mode not in {"row", "column"}:
            log(f"OVERSCAN.mode='{mode}' not in {{'row','column'}} – defaulting to 'row'.")
            mode = "row"

        log(
            f"=== OVERSCAN CORRECTION ===\n"
            f"Using region x={x1}:{x2}, y={y1}:{y2}, mode={mode}."
        )

        corrected_files: List[str] = []

        for fname in input_files:
            try:
                with fits.open(fname) as hdul:
                    data = hdul[0].data
                    if data is None or data.ndim != 2:
                        raise ValueError("Only 2D images are supported for overscan.")

                    data = np.asarray(data, dtype=np.float32)
                    ys, xs = data.shape
                    # Clamp region to image bounds
                    xx1 = max(0, min(xs, x1))
                    xx2 = max(0, min(xs, x2))
                    yy1 = max(0, min(ys, y1))
                    yy2 = max(0, min(ys, y2))

                    if xx1 >= xx2 or yy1 >= yy2:
                        raise ValueError("Overscan region is empty after clipping to image.")

                    overscan = data[yy1:yy2, xx1:xx2]

                    if mode == "row":
                        # median along columns -> one value per row
                        profile = np.median(overscan, axis=1, keepdims=True)
                    else:  # column mode
                        profile = np.median(overscan, axis=0, keepdims=True)

                    corrected = data - profile

                    out_name = self._add_suffix(fname, "-o")
                    hdul[0].data = corrected.astype(data.dtype)
                    hdul.writeto(out_name, overwrite=True)
                    corrected_files.append(out_name)

                    if verbose:
                        log(f"[overscan] Wrote {out_name}")
            except Exception as exc:  # noqa: BLE001
                log(f"[overscan] Failed for {fname}: {exc}")
                # fall back to original file if something went wrong
                corrected_files.append(fname)

        return corrected_files

    # ------------------------------------------------------------------
    # Cosmetic correction
    # ------------------------------------------------------------------
    def _apply_cosmetic_if_enabled(
        self,
        final_files: List[str],
        log: Callable[[str], None],
        verbose: bool,
    ) -> List[str]:
        cos_cfg = self.full_config.get("COSMETIC", {})
        enabled = self._as_bool(cos_cfg.get("enabled", False), False)
        if not enabled:
            return final_files

        bpm_path = cos_cfg.get("bad_pixel_mask")
        if not bpm_path:
            log("COSMETIC.enabled=True but bad_pixel_mask is not set – skipping cosmetic correction.")
            return final_files

        bpm_path = str(Path(bpm_path).expanduser().resolve())
        if not Path(bpm_path).is_file():
            log(f"COSMETIC.bad_pixel_mask file not found: {bpm_path} – skipping cosmetic correction.")
            return final_files

        try:
            mask_data = fits.getdata(bpm_path)
            if mask_data is None or mask_data.ndim != 2:
                raise ValueError("Bad pixel mask must be a 2D image.")
            bad_mask = np.asarray(mask_data) != 0
        except Exception as exc:  # noqa: BLE001
            log(f"Failed to read bad pixel mask {bpm_path}: {exc} – skipping cosmetic correction.")
            return final_files

        kernel = Box2DKernel(3)

        log("=== COSMETIC CORRECTION (bad pixels) ===")

        for fname in final_files:
            try:
                with fits.open(fname, mode="update") as hdul:
                    data = hdul[0].data
                    if data is None or data.ndim != 2:
                        continue
                    data = np.asarray(data, dtype=np.float32)

                    if bad_mask.shape != data.shape:
                        log(
                            f"[cosmetic] Mask shape {bad_mask.shape} != image shape {data.shape} "
                            f"for {fname} – skipping this file."
                        )
                        continue

                    smoothed = convolve(data, kernel, boundary="extend", normalize_kernel=True)
                    corrected = data.copy()
                    corrected[bad_mask] = smoothed[bad_mask]

                    hdul[0].data = corrected.astype(data.dtype)
                    hdul.flush()

                    if verbose:
                        log(f"[cosmetic] Updated {fname}")
            except Exception as exc:  # noqa: BLE001
                log(f"[cosmetic] Failed for {fname}: {exc}")

        return final_files

    # ------------------------------------------------------------------
    # Background modelling
    # ------------------------------------------------------------------
    def _apply_background_if_enabled(
        self,
        final_files: List[str],
        log: Callable[[str], None],
        verbose: bool,
    ) -> List[str]:
        bkg_cfg = self.full_config.get("BACKGROUND", {})
        enabled = self._as_bool(bkg_cfg.get("background_subtraction", False), False)
        if not enabled:
            return final_files

        box_size_raw = bkg_cfg.get("box_size", 64)
        try:
            box_size = max(3, int(box_size_raw))
        except Exception:
            box_size = 64

        log(
            "=== BACKGROUND MODELLING ===\n"
            f"Using Box2D kernel size = {box_size} pixels. "
            "Background is subtracted in-place from final images."
        )

        kernel = Box2DKernel(box_size)

        for fname in final_files:
            try:
                with fits.open(fname, mode="update") as hdul:
                    data = hdul[0].data
                    if data is None or data.ndim != 2:
                        continue
                    data = np.asarray(data, dtype=np.float32)

                    background = convolve(
                        data,
                        kernel,
                        boundary="extend",
                        normalize_kernel=True,
                    )
                    corrected = data - background

                    hdul[0].data = corrected.astype(data.dtype)
                    hdul.flush()

                    if verbose:
                        log(f"[background] Updated {fname}")
            except Exception as exc:  # noqa: BLE001
                log(f"[background] Failed for {fname}: {exc}")

        return final_files


# ----------------------------------------------------------------------
# Simple CLI entrypoint (console use)
# ----------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(
        description="Run full CCD calibration pipeline (console version)."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to LISNYKY_Moravian-C4-16000.ini",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-d",
        "--directory",
        help="Directory with raw FITS files",
    )
    group.add_argument(
        "-l",
        "--list",
        help="Text file (.lst/.txt) with FITS paths",
    )
    group.add_argument(
        "-a",
        "--archive",
        help="Archive (.zip/.tar/.tar.gz) with FITS files",
    )

    parser.add_argument(
        "-e",
        "--extra-calib",
        nargs="+",
        help="Extra calibration FITS files (long darks, extra flats, etc.)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose log messages",
    )

    args = parser.parse_args()
    pipeline = CalibrationPipeline(args.config)

    if args.directory:
        src = args.directory
        stype = "directory"
    elif args.list:
        src = args.list
        stype = "list"
    else:
        src = args.archive
        stype = "archive"

    verbose = not args.quiet
    pipeline.run_full_calibration(
        src,
        source_type=stype,
        extra_calib_files=args.extra_calib,
        verbose=verbose,
        log_callback=print,
    )


if __name__ == "__main__":
    _cli()