"""The source registry: what goes into the index, and on what terms.

sources.yaml (project root) lists every document the system can search:
the textbook, public guidelines and policy documents, clinic protocols.
Each entry says where the file is, which loader reads it, what it may be
used for, and how it is cited. ingest.py builds the index from the enabled
entries, and nothing that is not listed here gets indexed.

Why a registry rather than a folder scan: licensing. A file in data/raw
says nothing about whether it may be indexed or quoted to users. An entry
here has to state its license, so adding a source is a deliberate act.

Run:  python3 -m src.ingest --list      # status of every entry
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src import config

FORMATS = ("pdf_textbook", "pdf", "html", "nxml", "docx", "markdown")
DOMAINS = ("clinical", "operations")
TYPES = ("textbook", "guideline", "review", "policy", "protocol", "article")
FIELDS = {"id", "title", "label", "type", "domain", "format", "path", "license",
          "url", "year", "enabled", "notes", "options"}
REQUIRED = ("id", "title", "type", "domain", "format", "path", "license")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class RegistryError(ValueError):
    """sources.yaml is malformed. The message lists every problem at once."""


@dataclass
class Source:
    id: str
    title: str
    label: str          # short name used in citations: "Mohs textbook"
    type: str
    domain: str
    format: str
    path: Path          # absolute
    license: str
    url: str = ""
    year: Optional[int] = None
    enabled: bool = True
    notes: str = ""
    options: dict = field(default_factory=dict)

    @property
    def available(self):
        return self.path.exists()


def parse_registry(data, root=None):
    """Parsed YAML -> [Source]. Pure: no file access except path resolution.

    Collects every problem before raising, so one run of --list shows them all.
    """
    root = Path(root or config.ROOT)
    entries = data.get("sources") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise RegistryError("sources.yaml must have a top-level 'sources:' list")

    problems, sources, seen = [], [], set()
    for n, entry in enumerate(entries, start=1):
        where = f"entry {n}"
        if not isinstance(entry, dict):
            problems.append(f"{where}: not a mapping")
            continue
        where = f"entry {n} ({entry.get('id', 'no id')})"

        unknown = set(entry) - FIELDS
        if unknown:
            problems.append(f"{where}: unknown field(s) {sorted(unknown)} - typo?")
        missing = [k for k in REQUIRED if not entry.get(k)]
        if missing:
            problems.append(f"{where}: missing {missing}")
            continue

        sid = str(entry["id"])
        if not ID_RE.match(sid):
            problems.append(f"{where}: id must be lowercase letters, digits, underscores")
        if sid in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(sid)
        for key, allowed in (("format", FORMATS), ("domain", DOMAINS), ("type", TYPES)):
            if entry[key] not in allowed:
                problems.append(f"{where}: {key} '{entry[key]}' is not one of {list(allowed)}")
        year = entry.get("year")
        if year is not None and not isinstance(year, int):
            problems.append(f"{where}: year must be a number")
        options = entry.get("options") or {}
        if not isinstance(options, dict):
            problems.append(f"{where}: options must be a mapping")
            options = {}

        path = Path(str(entry["path"]))
        sources.append(Source(
            id=sid,
            title=str(entry["title"]),
            label=str(entry.get("label") or entry["title"]),
            type=entry["type"],
            domain=entry["domain"],
            format=entry["format"],
            path=path if path.is_absolute() else root / path,
            license=str(entry["license"]),
            url=str(entry.get("url") or ""),
            year=year if isinstance(year, int) else None,
            enabled=bool(entry.get("enabled", True)),
            notes=str(entry.get("notes") or "").strip(),
            options=options,
        ))

    if problems:
        raise RegistryError("sources.yaml has problems:\n  - " + "\n  - ".join(problems))
    return sources


def load_registry(path=None):
    """Read and validate sources.yaml. PyYAML is imported here, not at module
    level, because the app never needs it - only ingestion does."""
    import yaml
    path = Path(path or config.SOURCES_PATH)
    return parse_registry(yaml.safe_load(path.read_text()), root=config.ROOT)


def enabled_sources(sources):
    """The enabled entries. An enabled entry whose file is missing is an
    error, and the message carries its download notes."""
    chosen = [s for s in sources if s.enabled]
    missing = [s for s in chosen if not s.available]
    if missing:
        lines = []
        for s in missing:
            lines.append(f"{s.id}: file not found at {s.path}")
            if s.url:
                lines.append(f"    get it from: {s.url}")
            if s.notes:
                lines.append(f"    notes: {s.notes}")
        raise RegistryError("enabled source(s) missing:\n  " + "\n  ".join(lines)
                            + "\nDownload them, or set enabled: false.")
    return chosen
