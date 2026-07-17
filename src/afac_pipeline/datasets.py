"""Dataset discovery, inspection, and extraction utilities."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    file_name: str
    source: str
    read_bytes: Callable[[], bytes]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    zip_glob: str
    dir_glob: str
    image_prefixes: tuple[str, ...]
    md_prefixes: tuple[str, ...] = ()
    mapping_names: tuple[str, ...] = ()


TRAIN_SPEC = DatasetSpec(
    name="train",
    zip_glob="AFAC 训练数据集.zip",
    dir_glob="AFAC 训练数据集",
    image_prefixes=(
        "finixdocbench_huge_long_100/images/",
        "finixdocbench_huge_table_100/images/",
    ),
    md_prefixes=(
        "finixdocbench_huge_long_100/mds/",
        "finixdocbench_huge_table_100/mds/",
    ),
    mapping_names=(
        "finixdocbench_huge_long_100/id_mapping.csv",
        "finixdocbench_huge_table_100/id_mapping.csv",
    ),
)
A_SPEC = DatasetSpec(
    name="a",
    zip_glob="AFAC A榜评测数据集*.zip",
    dir_glob="AFAC A榜评测数据集*",
    image_prefixes=(
        "finix_huge_long_rest_A/images/",
        "finix_huge_table_rest_A/images/",
    ),
)
MOCK_SUBMISSION_ZIP = "finix_ab_A_submit_mock.csv.zip"
MOCK_SUBMISSION_NAME = "finix_ab_A_submit_mock.csv"


def get_dataset_spec(name: str) -> DatasetSpec:
    if name == "train":
        return TRAIN_SPEC
    if name == "a":
        return A_SPEC
    raise ValueError(f"Unknown dataset {name!r}; expected 'train' or 'a'")


def iter_dataset_images(raw_dir: Path, dataset: str) -> Iterator[ImageRecord]:
    spec = get_dataset_spec(dataset)
    dir_roots = _dataset_dir_roots(raw_dir, spec.dir_glob)
    if dir_roots:
        yield from _iter_dir_images(dir_roots)
        return
    yield from iter_zip_images(raw_dir, spec.zip_glob, spec.image_prefixes)


def iter_zip_images(
    raw_dir: Path,
    zip_glob: str,
    image_prefixes: tuple[str, ...],
) -> Iterator[ImageRecord]:
    zip_paths = sorted(raw_dir.glob(zip_glob))
    seen: set[str] = set()
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            image_names = [
                name
                for name in zf.namelist()
                if _is_data_image(name, image_prefixes)
            ]
        for member_name in sorted(image_names):
            file_name = Path(member_name).name
            if file_name in seen:
                continue
            seen.add(file_name)

            def read_bytes(
                zip_path: Path = zip_path,
                member_name: str = member_name,
            ) -> bytes:
                with zipfile.ZipFile(zip_path) as zf:
                    return zf.read(member_name)

            yield ImageRecord(
                file_name=file_name,
                source=f"{zip_path.name}:{member_name}",
                read_bytes=read_bytes,
            )


def iter_dir_images(input_dir: Path) -> Iterator[ImageRecord]:
    yield from _iter_dir_images([input_dir])


def inspect_raw_data(raw_dir: Path) -> dict[str, object]:
    train_zip = raw_dir / TRAIN_SPEC.zip_glob
    a_zips = sorted(raw_dir.glob(A_SPEC.zip_glob))
    mock_zip = raw_dir / MOCK_SUBMISSION_ZIP
    mock_csv = raw_dir / MOCK_SUBMISSION_NAME
    mock_source = mock_submission_source(raw_dir)
    train_dir_roots = _dataset_dir_roots(raw_dir, TRAIN_SPEC.dir_glob)
    a_dir_roots = _dataset_dir_roots(raw_dir, A_SPEC.dir_glob)

    summary: dict[str, object] = {
        "raw_dir": str(raw_dir),
        "train_zip_exists": train_zip.exists(),
        "a_zip_files": [path.name for path in a_zips],
        "mock_zip_exists": mock_zip.exists(),
        "mock_csv_exists": mock_csv.exists(),
        "train_dir_roots": [path.name for path in train_dir_roots],
        "a_dir_roots": [path.name for path in a_dir_roots],
    }
    if train_zip.exists():
        summary.update(_count_zip_dataset(train_zip, TRAIN_SPEC))
    elif train_dir_roots:
        summary.update(_count_dir_dataset(train_dir_roots, TRAIN_SPEC))
    if a_zips:
        a_images = list(iter_dataset_images(raw_dir, "a"))
        summary["a_images"] = len(a_images)
    elif a_dir_roots:
        summary["a_images"] = len(list(_iter_dir_images(a_dir_roots)))
    if mock_source:
        mock_rows = read_mock_submission(mock_source)
        summary["mock_rows"] = len(mock_rows)
        summary["mock_columns"] = list(mock_rows[0].keys()) if mock_rows else []
        if a_zips or a_dir_roots:
            a_names = {record.file_name for record in iter_dataset_images(raw_dir, "a")}
            mock_names = {row["file_name"] for row in mock_rows}
            summary["a_mock_name_intersection"] = len(a_names & mock_names)
            summary["mock_names_missing_from_a"] = len(mock_names - a_names)
    return summary


def read_mock_submission(mock_zip: Path) -> list[dict[str, str]]:
    if mock_zip.suffix.lower() == ".csv":
        text = mock_zip.read_text(encoding="utf-8-sig")
    else:
        with zipfile.ZipFile(mock_zip) as zf:
            text = zf.read(MOCK_SUBMISSION_NAME).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def extract_dataset(raw_dir: Path, output_dir: Path, dataset: str) -> list[Path]:
    if dataset == "mock":
        zip_paths = [raw_dir / MOCK_SUBMISSION_ZIP]
    else:
        spec = get_dataset_spec(dataset)
        zip_paths = sorted(raw_dir.glob(spec.zip_glob))

    extracted: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_paths:
        if not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                if _should_skip_member(member.filename):
                    continue
                target = output_dir / member.filename
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
                extracted.append(target)
    return extracted


def dataset_file_names(records: Iterable[ImageRecord]) -> list[str]:
    return [record.file_name for record in records]


def _count_zip_dataset(zip_path: Path, spec: DatasetSpec) -> dict[str, int]:
    counts = {"train_images": 0, "train_mds": 0, "train_mapping_rows": 0}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        counts["train_images"] = sum(_is_data_image(name, spec.image_prefixes) for name in names)
        if spec.md_prefixes:
            counts["train_mds"] = sum(_is_markdown(name, spec.md_prefixes) for name in names)
        for mapping_name in spec.mapping_names:
            if mapping_name in names:
                text = zf.read(mapping_name).decode("utf-8-sig")
                counts["train_mapping_rows"] += max(0, sum(1 for _ in csv.DictReader(io.StringIO(text))))
    return counts


def _count_dir_dataset(dir_roots: Iterable[Path], spec: DatasetSpec) -> dict[str, int]:
    counts = {"train_images": 0, "train_mds": 0, "train_mapping_rows": 0}
    image_files = 0
    md_files = 0
    mapping_rows = 0
    for root in dir_roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name == ".DS_Store" or path.name.startswith("._") or "__MACOSX" in path.parts:
                continue
            if path.suffix.lower() in IMAGE_SUFFIXES:
                image_files += 1
            relative_path = path.relative_to(root).as_posix()
            if _is_markdown(relative_path, spec.md_prefixes):
                md_files += 1
            if relative_path in spec.mapping_names:
                text = path.read_text(encoding="utf-8-sig")
                mapping_rows += max(0, sum(1 for _ in csv.DictReader(io.StringIO(text))))
    counts["train_images"] = image_files
    counts["train_mds"] = md_files
    counts["train_mapping_rows"] = mapping_rows
    return counts


def _dataset_dir_roots(raw_dir: Path, dir_glob: str) -> list[Path]:
    return sorted(path for path in raw_dir.glob(dir_glob) if path.is_dir())


def mock_submission_source(raw_dir: Path) -> Path | None:
    csv_path = raw_dir / MOCK_SUBMISSION_NAME
    if csv_path.exists():
        return csv_path
    zip_path = raw_dir / MOCK_SUBMISSION_ZIP
    if zip_path.exists():
        return zip_path
    return None


def _iter_dir_images(dir_roots: Iterable[Path]) -> Iterator[ImageRecord]:
    seen: set[str] = set()
    image_paths = sorted(
        path
        for root in dir_roots
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.name != ".DS_Store"
        and not path.name.startswith("._")
        and "__MACOSX" not in path.parts
    )
    for image_path in image_paths:
        file_name = image_path.name
        if file_name in seen:
            continue
        seen.add(file_name)
        yield ImageRecord(
            file_name=file_name,
            source=str(image_path),
            read_bytes=image_path.read_bytes,
        )


def _is_data_image(member_name: str, image_prefixes: tuple[str, ...]) -> bool:
    path = Path(member_name)
    return (
        any(member_name.startswith(prefix) for prefix in image_prefixes)
        and not member_name.endswith("/")
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not path.name.startswith("._")
        and "__MACOSX" not in path.parts
    )


def _is_markdown(member_name: str, md_prefixes: tuple[str, ...]) -> bool:
    path = Path(member_name)
    return (
        any(member_name.startswith(prefix) for prefix in md_prefixes)
        and not member_name.endswith("/")
        and path.suffix.lower() == ".md"
        and not path.name.startswith("._")
        and "__MACOSX" not in path.parts
    )


def _should_skip_member(member_name: str) -> bool:
    path = Path(member_name)
    return (
        "__MACOSX" in path.parts
        or path.name == ".DS_Store"
        or path.name.startswith("._")
    )
