from __future__ import annotations

from pathlib import Path
from typing import Iterable


RASTER_DPI = 300
OUTPUT_SUFFIXES = (".png", ".pdf")
LEGACY_IMAGE_SUFFIXES = {".svg", ".png", ".pdf", ".jpg", ".jpeg"}


def output_prefix(path: Path) -> Path:
    if path.suffix.lower() in LEGACY_IMAGE_SUFFIXES:
        return path.with_suffix("")
    return path


def output_paths(path: Path) -> tuple[Path, Path]:
    prefix = output_prefix(path)
    return prefix.with_suffix(".png"), prefix.with_suffix(".pdf")


def save_png_pdf(fig, path: Path, *, dpi: int = RASTER_DPI) -> tuple[Path, Path]:
    png_path, pdf_path = output_paths(path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, format="png", dpi=dpi)
    fig.savefig(pdf_path, format="pdf")
    return png_path, pdf_path


def format_written(paths: Iterable[Path]) -> str:
    return ", ".join(str(path) for path in paths)
