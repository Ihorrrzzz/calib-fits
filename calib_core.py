# calib_core.py

from __future__ import annotations

import argparse
import tarfile
import zipfile
import shutil
from pathlib import Path
from typing import Callable, Optional, List

from calib_config import CalibConfig
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

      RAW → master bias → bias-sub
          → master dark → dark-sub
          → master flats → flat-sub (science)
    """

    FITS_EXTENSIONS = {".fits", ".fit", ".FITS", ".FIT"}

    def __init__(self, config_path: str):
        self.config_path = str(config_path)
        self.cfg = CalibConfig(self.config_path)
        self.full_config = self.cfg.config

        self.working_dir = Path(self.cfg.get("DATA_STRUCTURE", "working_dir", "./work"))
        self.results_dir = Path(self.cfg.get("DATA_STRUCTURE", "results_dir", "./results"))
        self.results_aux_dir = Path(
            self.cfg.get("DATA_STRUCTURE", "results_aux_dir", "./results/aux")
        )

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
        verbose: bool = True,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        """
        Run the full calibration pipeline.

        raw_source:
          - if source_type == "directory": path to directory with FITS
          - if source_type == "list": path to txt/lst file with FITS paths
          - if source_type == "archive": .zip/.tar(.gz) with FITS

        Returns list of final calibrated OBJECT frames (paths).
        """
        log = log_callback or print

        input_files = self._collect_input_files(raw_source, source_type, log)
        if not input_files:
            raise RuntimeError("No FITS files found for calibration.")

        img_proc_cfg = self.full_config.get("IMAGE_PROCESSING", {})

        # ------------------ Bias --------------------------------------
        bias_enabled = img_proc_cfg.get("bias_subtraction", True)
        if bias_enabled:
            log("=== CREATING MASTER BIAS ===")
            masterbias_path = mkmasterbias.create_master_bias(
                input_files,
                self.cfg,
                output_filename="masterbias.fits",
                method=img_proc_cfg.get("bias_subtraction_method"),
                sigma=img_proc_cfg.get("bias_subtraction_sigma", 2.5),
                make_png=False,
                verbose=verbose,
            )

            log("=== BIAS CORRECTION ===")
            bias_corrected_files = bias_correction.apply_bias_correction(
                input_files,
                masterbias_path,
                self.cfg,
                verbose=verbose,
            )
        else:
            log("Bias subtraction disabled in config; skipping bias steps.")
            bias_corrected_files = list(input_files)
            masterbias_path = None  # noqa: F841 (might later be used)

        # ------------------ Dark --------------------------------------
        dark_enabled = img_proc_cfg.get("dark_correction", True)
        if dark_enabled:
            log("=== CREATING MASTER DARK ===")
            masterdark_path = mkmasterdark.make_master_dark(
                bias_corrected_files,
                self.cfg,
                output_filename="masterdark.fits",
                method=img_proc_cfg.get("dark_correction_method"),
                make_png=False,
                verbose=verbose,
            )

            log("=== DARK CORRECTION ===")
            dark_corrected_files = dark_correction.apply_dark_correction(
                bias_corrected_files,
                masterdark_path,
                self.cfg,
                verbose=verbose,
            )
        else:
            log("Dark correction disabled in config; skipping dark steps.")
            dark_corrected_files = list(bias_corrected_files)
            masterdark_path = None  # noqa: F841

        # ------------------ Flats & science ---------------------------
        flat_enabled = img_proc_cfg.get("flat_correction", True)
        if flat_enabled:
            log("=== CREATING MASTER FLATS ===")
            masterflats = mkmasterflats.create_master_flats(
                dark_corrected_files,
                self.cfg,
                verbose=verbose,
            )

            if verbose:
                for filt, (mf, mfn) in masterflats.items():
                    log(f"Filter {filt}: masterflat={mf}, masterflat_norm={mfn}")

            log("=== FLAT CORRECTION (SCIENCE FRAMES) ===")
            final_files = flat_correction.apply_flat_correction(
                dark_corrected_files,
                self.cfg,
                verbose=verbose,
            )
        else:
            log("Flat correction disabled in config; skipping flats.")
            final_files = list(dark_corrected_files)

        log("=== CALIBRATION COMPLETE ===")
        log(f"Final calibrated science images are in: {self.results_dir}")
        return final_files

    # ------------------------------------------------------------------
    # Internal helpers
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
                str(f)
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
            files = []
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
        verbose=verbose,
        log_callback=print,
    )


if __name__ == "__main__":
    _cli()
