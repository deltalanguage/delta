"""
Error types and error reporting for Delta compiler.

Provides a hierarchy of error types with rich source location
information and formatted error messages.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from enum import Enum, auto

if TYPE_CHECKING:
    from delta.source import SourceLocation, SourceFile


class ErrorSeverity(Enum):
    """Severity levels for diagnostics."""
    ERROR = auto()
    WARNING = auto()
    NOTE = auto()
    HINT = auto()


class ErrorCode(Enum):
    """
    Enumerated error codes for all compiler errors.
    
    Format: E{category}{number}
    Categories:
      - 0xx: Lexer errors
      - 1xx: Parser errors
      - 2xx: Name resolution errors
      - 3xx: Type errors
      - 4xx: Effect errors
      - 5xx: Mode errors
      - 6xx: Constraint errors
      - 7xx: Backend errors
      - 8xx: Runtime errors
    """
    # Lexer errors (0xx)
    E001_UNEXPECTED_CHARACTER = "E001"
    E002_UNTERMINATED_STRING = "E002"
    E003_INVALID_NUMBER = "E003"
    E004_UNTERMINATED_COMMENT = "E004"
    
    # Parser errors (1xx)
    E101_UNEXPECTED_TOKEN = "E101"
    E102_EXPECTED_EXPRESSION = "E102"
    E103_EXPECTED_TYPE = "E103"
    E104_EXPECTED_IDENTIFIER = "E104"
    E105_UNCLOSED_DELIMITER = "E105"
    E106_INVALID_SYNTAX = "E106"
    E107_EXPECTED_BLOCK = "E107"
    
    # Name resolution errors (2xx)
    E201_UNDEFINED_VARIABLE = "E201"
    E202_UNDEFINED_TYPE = "E202"
    E203_UNDEFINED_FUNCTION = "E203"
    E204_DUPLICATE_DEFINITION = "E204"
    E205_IMPORT_NOT_FOUND = "E205"
    E206_CIRCULAR_IMPORT = "E206"
    
    # Type errors (3xx)
    E301_TYPE_MISMATCH = "E301"
    E302_ARITY_MISMATCH = "E302"
    E303_NOT_CALLABLE = "E303"
    E304_NOT_INDEXABLE = "E304"
    E305_MISSING_FIELD = "E305"
    E306_INFINITE_TYPE = "E306"
    E307_AMBIGUOUS_TYPE = "E307"
    E308_INCOMPATIBLE_SHAPES = "E308"
    E309_INVALID_OPERATION = "E309"
    
    # Effect errors (4xx)
    E401_FORBIDDEN_EFFECT = "E401"
    E402_UNHANDLED_EFFECT = "E402"
    E403_EFFECT_LEAK = "E403"
    
    # Mode errors (5xx)
    E501_STOCHASTIC_IN_INFER = "E501"
    E502_GRADIENT_INTO_OBS = "E502"
    E503_NON_DIFF_IN_LEARN = "E503"
    E504_HARD_BRANCH_IN_TRAIN = "E504"
    E505_MODE_MISMATCH = "E505"
    E506_WHILE_IN_DIFF = "E506"
    
    # Constraint errors (6xx)
    E601_INVALID_CONSTRAINT = "E601"
    E602_UNSATISFIABLE = "E602"
    E603_CONSTRAINT_TYPE_ERROR = "E603"
    
    # Backend errors (7xx)
    E701_LOWERING_FAILED = "E701"
    E702_UNSUPPORTED_OP = "E702"
    E703_COMPILATION_FAILED = "E703"
    
    # Runtime errors (8xx)
    E801_EXECUTION_FAILED = "E801"
    E802_SHAPE_ERROR = "E802"
    E803_DEVICE_ERROR = "E803"


@dataclass
class RelatedInfo:
    """Additional information related to an error."""
    message: str
    location: Optional[SourceLocation] = None


@dataclass
class DeltaError(Exception):
    """
    Base class for all Delta compiler errors.
    
    Provides rich error information including source location,
    error code, and related information for context.
    """
    message: str
    location: Optional[SourceLocation] = None
    code: Optional[ErrorCode] = None
    severity: ErrorSeverity = ErrorSeverity.ERROR
    related: list[RelatedInfo] = field(default_factory=list)
    source: Optional[SourceFile] = None
    
    def __str__(self) -> str:
        return self.format()
    
    def format(self, show_source: bool = True) -> str:
        """Format the error for display."""
        parts = []
        
        # Severity and code
        severity_str = self.severity.name.lower()
        code_str = f"[{self.code.value}]" if self.code else ""
        
        # Location
        if self.location:
            loc_str = str(self.location)
            parts.append(f"{severity_str}{code_str}: {loc_str}")
        else:
            parts.append(f"{severity_str}{code_str}")
        
        # Message
        parts.append(f"  {self.message}")
        
        # Source snippet
        if show_source and self.source and self.location:
            parts.append("")
            parts.append(self.source.get_snippet(self.location))
        
        # Related information
        for info in self.related:
            parts.append("")
            if info.location:
                parts.append(f"note: {info.location}")
            parts.append(f"  {info.message}")
        
        return "\n".join(parts)
    
    def with_related(self, message: str, location: Optional[SourceLocation] = None) -> DeltaError:
        """Add related information to this error."""
        self.related.append(RelatedInfo(message=message, location=location))
        return self
    
    def with_source(self, source: SourceFile) -> DeltaError:
        """Attach source file for snippet display."""
        self.source = source
        return self


class LexerError(DeltaError):
    """Error during lexical analysis."""
    pass


class ParseError(DeltaError):
    """Error during parsing."""
    pass


class NameError(DeltaError):
    """Error during name resolution."""
    pass


class TypeError(DeltaError):
    """Error during type checking/inference."""
    pass


class EffectError(DeltaError):
    """Error related to effect tracking."""
    pass


class ModeError(DeltaError):
    """Error related to mode (train/infer/analyze) checking."""
    pass


class ConstraintError(DeltaError):
    """Error related to constraint compilation."""
    pass


class BackendError(DeltaError):
    """Error during backend lowering/compilation."""
    pass


class RuntimeError(DeltaError):
    """Error during execution."""
    pass


class CompileError(DeltaError):
    """General compilation error."""
    pass


@dataclass
class ErrorCollector:
    """
    Collects multiple errors during compilation.
    
    Allows compilation to continue after errors for better
    error reporting, while tracking whether any errors occurred.
    """
    errors: list[DeltaError] = field(default_factory=list)
    max_errors: int = 100
    
    def add(self, error: DeltaError) -> None:
        """Add an error to the collection."""
        if len(self.errors) < self.max_errors:
            self.errors.append(error)
    
    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return any(e.severity == ErrorSeverity.ERROR for e in self.errors)
    
    def error_count(self) -> int:
        """Count errors (not warnings/notes)."""
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.ERROR)
    
    def warning_count(self) -> int:
        """Count warnings."""
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.WARNING)
    
    def clear(self) -> None:
        """Clear all collected errors."""
        self.errors.clear()
    
    def raise_if_errors(self) -> None:
        """Raise a combined error if any errors were collected."""
        if self.has_errors():
            raise CompileError(
                message=f"Compilation failed with {self.error_count()} error(s)",
                related=[RelatedInfo(e.message, e.location) for e in self.errors[:5]]
            )
    
    def format_all(self, show_source: bool = True) -> str:
        """Format all errors for display."""
        parts = []
        for error in self.errors:
            parts.append(error.format(show_source))
            parts.append("")
        
        summary = f"\n{self.error_count()} error(s), {self.warning_count()} warning(s)"
        parts.append(summary)
        
        return "\n".join(parts)
