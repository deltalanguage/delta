"""
Source file handling for Delta compiler.

Provides source location tracking, source file management, and 
utilities for error reporting with precise locations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator
import hashlib


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """
    Represents a precise location in source code.
    
    All positions are 1-indexed for user-facing display.
    """
    file: str
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    
    def __str__(self) -> str:
        if self.end_line is not None and self.end_line != self.line:
            return f"{self.file}:{self.line}:{self.column}-{self.end_line}:{self.end_column}"
        elif self.end_column is not None:
            return f"{self.file}:{self.line}:{self.column}-{self.end_column}"
        return f"{self.file}:{self.line}:{self.column}"
    
    def extend_to(self, other: SourceLocation) -> SourceLocation:
        """Create a new location spanning from this location to another."""
        return SourceLocation(
            file=self.file,
            line=self.line,
            column=self.column,
            end_line=other.end_line or other.line,
            end_column=other.end_column or other.column,
        )
    
    @classmethod
    def unknown(cls) -> SourceLocation:
        """Create a location for compiler-generated code."""
        return cls(file="<generated>", line=0, column=0)
    
    @classmethod
    def builtin(cls) -> SourceLocation:
        """Create a location for built-in definitions."""
        return cls(file="<builtin>", line=0, column=0)


@dataclass
class SourceFile:
    """
    Represents a Delta source file with content and metadata.
    
    Provides efficient line/column lookup and snippet extraction
    for error reporting.
    """
    path: str
    content: str
    _line_offsets: list[int] = field(default_factory=list, repr=False)
    _hash: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        self._compute_line_offsets()
    
    def _compute_line_offsets(self) -> None:
        """Compute byte offsets for the start of each line."""
        self._line_offsets = [0]
        for i, char in enumerate(self.content):
            if char == '\n':
                self._line_offsets.append(i + 1)
    
    @classmethod
    def from_path(cls, path: Path | str) -> SourceFile:
        """Load a source file from disk."""
        path = Path(path)
        content = path.read_text(encoding='utf-8')
        return cls(path=str(path), content=content)
    
    @classmethod
    def from_string(cls, content: str, name: str = "<string>") -> SourceFile:
        """Create a source file from a string (for REPL/testing)."""
        return cls(path=name, content=content)
    
    @property
    def hash(self) -> str:
        """Get content hash for caching."""
        if self._hash is None:
            self._hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        return self._hash
    
    @property
    def line_count(self) -> int:
        """Get total number of lines."""
        return len(self._line_offsets)
    
    def get_line(self, line_number: int) -> str:
        """Get a specific line (1-indexed)."""
        if line_number < 1 or line_number > self.line_count:
            return ""
        
        start = self._line_offsets[line_number - 1]
        if line_number < self.line_count:
            end = self._line_offsets[line_number] - 1  # Exclude newline
        else:
            end = len(self.content)
        
        return self.content[start:end]
    
    def get_lines(self, start_line: int, end_line: int) -> list[str]:
        """Get a range of lines (1-indexed, inclusive)."""
        return [self.get_line(i) for i in range(start_line, end_line + 1)]
    
    def offset_to_location(self, offset: int) -> SourceLocation:
        """Convert a byte offset to line/column location."""
        if offset < 0:
            offset = 0
        if offset >= len(self.content):
            offset = len(self.content) - 1 if self.content else 0
        
        # Binary search for line
        line = 1
        for i, line_offset in enumerate(self._line_offsets):
            if line_offset > offset:
                break
            line = i + 1
        
        column = offset - self._line_offsets[line - 1] + 1
        return SourceLocation(file=self.path, line=line, column=column)
    
    def location_to_offset(self, loc: SourceLocation) -> int:
        """Convert a line/column location to byte offset."""
        if loc.line < 1 or loc.line > self.line_count:
            return 0
        return self._line_offsets[loc.line - 1] + loc.column - 1
    
    def get_snippet(self, loc: SourceLocation, context_lines: int = 2) -> str:
        """
        Get a code snippet around a location for error display.
        
        Returns a formatted string with line numbers and a caret
        pointing to the error location.
        """
        start = max(1, loc.line - context_lines)
        end = min(self.line_count, (loc.end_line or loc.line) + context_lines)
        
        lines = []
        max_line_num_width = len(str(end))
        
        for i in range(start, end + 1):
            line_content = self.get_line(i)
            line_num = str(i).rjust(max_line_num_width)
            
            # Determine if this line is part of the error span
            is_error_line = loc.line <= i <= (loc.end_line or loc.line)
            marker = ">" if is_error_line else " "
            
            lines.append(f"{marker} {line_num} | {line_content}")
            
            # Add caret line for the primary error location
            if i == loc.line:
                padding = " " * (max_line_num_width + 4 + loc.column - 1)
                if loc.end_column and loc.end_line == loc.line:
                    underline = "^" * (loc.end_column - loc.column + 1)
                else:
                    underline = "^"
                lines.append(f"{padding}{underline}")
        
        return "\n".join(lines)
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over lines."""
        for i in range(1, self.line_count + 1):
            yield self.get_line(i)
    
    def __len__(self) -> int:
        """Return content length in bytes."""
        return len(self.content)


@dataclass
class SourceSpan:
    """
    A span of source code with the actual text content.
    
    Used for detailed error messages and source-level debugging.
    """
    location: SourceLocation
    text: str
    source: SourceFile
    
    @classmethod
    def from_location(cls, source: SourceFile, loc: SourceLocation) -> SourceSpan:
        """Create a span from a source file and location."""
        start_offset = source.location_to_offset(loc)
        if loc.end_line and loc.end_column:
            end_loc = SourceLocation(
                file=loc.file,
                line=loc.end_line,
                column=loc.end_column
            )
            end_offset = source.location_to_offset(end_loc)
        else:
            end_offset = start_offset + 1
        
        text = source.content[start_offset:end_offset]
        return cls(location=loc, text=text, source=source)
