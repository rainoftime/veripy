"""Configuration classes for CLI verification."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class VerificationConfig:
    """Configuration for verification runs."""
    files: List[Path] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    output_format: str = "text"
    show_counterexample: bool = True
    show_statistics: bool = True
    strict: bool = False
    timeout: Optional[int] = None
    solver: str = "z3"
    incremental: bool = True
    cache: bool = True
    workers: int = 1
