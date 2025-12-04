# calibration/calib_prep_lists.py

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List

SUFFIXES = ["", "-b", "-d", "-bd", "-bf", "-df", "-bdf"]
FITS_EXTENSIONS = {".fits", ".fit", ".FITS", ".FIT"}


# ----------------------------------------------------------------------
# Helpers used both by CLI and programmatically
# ----------------------------------------------------------------------
def fits_files_in_directory(directory: Path) -> List[str]:
    """Return absolute paths of FITS files in *directory* (non-recursive)."""
    return sorted(
        str(p.resolve())
        for p in directory.iterdir()
        if p.is_file() and p.suffix in FITS_EXTENSIONS
    )


def modified_filename(original: str, suffix: str) -> str:
    """Return *original* with *suffix* inserted before the extension."""
    p = Path(original)
    return str(p.with_name(p.stem + suffix + p.suffix))


def read_list_file(listfile: Path) -> List[str]:
    """Read newline-separated filenames from *listfile* (blank lines skipped)."""
    with listfile.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_list_file(name: str, lines: Iterable[str]) -> None:
    """Write *lines* to *name* in the current working directory."""
    with Path(name).open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_lists(base_name: str, originals: List[str]) -> None:
    """
    Given *originals* (absolute filenames), write base and derivative list files:
      base.lst, base-b.lst, base-d.lst, base-bd.lst,
      base-bf.lst, base-df.lst, base-bdf.lst
    """
    if not Path(base_name + ".lst").exists():
        write_list_file(f"{base_name}.lst", originals)

    for suffix in SUFFIXES[1:]:
        derived = [modified_filename(fn, suffix) for fn in originals]
        write_list_file(f"{base_name}{suffix}.lst", derived)


# ----------------------------------------------------------------------
# Programmatic API (no sys.exit here)
# ----------------------------------------------------------------------
def build_lists_from_directory(directory: str) -> str:
    """
    Build .lst files from a directory with FITS files.

    Returns: path to base list file (e.g. '<dirname>.lst').
    """
    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory '{dir_path}' not found")

    originals = fits_files_in_directory(dir_path)
    if not originals:
        raise ValueError(f"No FITS files found in directory '{dir_path}'")

    base_name = dir_path.name
    generate_lists(base_name, originals)
    return f"{base_name}.lst"


def build_lists_from_list(listfile: str) -> str:
    """
    Build derivative .lst files from an existing list of filenames.

    Returns: path to base list file (same as listfile, normalised to '<stem>.lst').
    """
    list_path = Path(listfile).expanduser().resolve()
    if not list_path.is_file():
        raise FileNotFoundError(f"List file '{list_path}' not found")

    originals = read_list_file(list_path)
    originals = [str(Path(p).expanduser().resolve()) for p in originals]
    base_name = list_path.with_suffix("").name
    generate_lists(base_name, originals)
    return f"{base_name}.lst"


def build_lists_from_archive(archive: str) -> str:
    """
    Build .lst files from an archive (zip/tar/tar.gz) containing FITS files.

    Returns: path to base list file (e.g. '<archive_name>.lst').
    """
    arc_path = Path(archive).expanduser().resolve()
    if not arc_path.is_file():
        raise FileNotFoundError(f"Archive '{arc_path}' not found")

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        if zipfile.is_zipfile(arc_path):
            with zipfile.ZipFile(arc_path, "r") as zf:
                zf.extractall(tmpdir)
        elif tarfile.is_tarfile(arc_path):
            with tarfile.open(arc_path, "r:*") as tf:
                tf.extractall(tmpdir)
        else:
            raise ValueError("Unsupported archive format")

        originals = [
            str(p.resolve())
            for p in tmpdir.rglob("*")
            if p.is_file() and p.suffix in FITS_EXTENSIONS
        ]
        if not originals:
            raise ValueError("No FITS files found inside archive")

        base_name = arc_path.stem.replace(".tar", "")
        generate_lists(base_name, sorted(originals))
        return f"{base_name}.lst"


# ----------------------------------------------------------------------
# CLI (optional)
# ----------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare calibration file lists for FITS image processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-l", "--list", metavar="FILE", help="existing list file with FITS names")
    group.add_argument("-d", "--directory", metavar="DIR", help="directory containing FITS files")
    group.add_argument("-a", "--archive", metavar="ARCH", help="zip, tar, or tar.gz archive")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        if args.list:
            build_lists_from_list(args.list)
        elif args.directory:
            build_lists_from_directory(args.directory)
        elif args.archive:
            build_lists_from_archive(args.archive)
        else:
            raise RuntimeError("No valid input source specified")
    except Exception as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
