"""
Token definitions for Delta lexer.

Defines all token types used in the Delta language, including
keywords, operators, and special tokens for differentiable constructs.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Any

from delta.source import SourceLocation


class TokenType(Enum):
    """
    All token types in Delta.
    
    Organized by category for maintainability.
    """
    # Literals
    INT_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    BOOL_LITERAL = auto()
    
    # Identifiers
    IDENTIFIER = auto()
    
    # Keywords - Control Flow
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    
    # Keywords - Definitions
    FN = auto()
    LET = auto()
    MUT = auto()
    TYPE = auto()
    STRUCT = auto()
    ENUM = auto()
    IMPL = auto()
    TRAIT = auto()
    
    # Keywords - Delta-specific
    PARAM = auto()          # Learnable parameter
    OBS = auto()            # Observed/fixed value
    CONST = auto()          # Compile-time constant
    LEARN = auto()          # Learning block
    CONSTRAINT = auto()     # Constraint definition
    REQUIRE = auto()        # Hard constraint
    PREFER = auto()         # Soft constraint
    TEMPERATURE = auto()    # Temperature annotation
    NON_DIFF = auto()       # Non-differentiable block
    RAND = auto()           # Random variable
    OBSERVE = auto()        # Probabilistic observation
    WHERE = auto()          # Constraint clause
    
    # Keywords - Mode
    TRAIN = auto()
    INFER = auto()
    ANALYZE = auto()
    
    # Keywords - Type
    TENSOR = auto()
    SCALAR = auto()
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    SHAPE = auto()
    
    # Keywords - Other
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Operators - Arithmetic
    PLUS = auto()           # +
    MINUS = auto()          # -
    STAR = auto()           # *
    SLASH = auto()          # /
    PERCENT = auto()        # %
    DOUBLE_STAR = auto()    # **
    AT = auto()             # @ (matrix multiply)
    
    # Operators - Comparison
    EQ = auto()             # ==
    NE = auto()             # !=
    LT = auto()             # <
    LE = auto()             # <=
    GT = auto()             # >
    GE = auto()             # >=
    
    # Operators - Assignment
    ASSIGN = auto()         # =
    PLUS_ASSIGN = auto()    # +=
    MINUS_ASSIGN = auto()   # -=
    STAR_ASSIGN = auto()    # *=
    SLASH_ASSIGN = auto()   # /=
    
    # Operators - Other
    ARROW = auto()          # ->
    FAT_ARROW = auto()      # =>
    DOUBLE_COLON = auto()   # ::
    PIPE = auto()           # |
    AMPERSAND = auto()      # &
    CARET = auto()          # ^
    TILDE = auto()          # ~
    DOT = auto()            # .
    DOUBLE_DOT = auto()     # ..
    QUESTION = auto()       # ?
    BANG = auto()           # !
    
    # Delimiters
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    LBRACKET = auto()       # [
    RBRACKET = auto()       # ]
    LBRACE = auto()         # {
    RBRACE = auto()         # }
    COMMA = auto()          # ,
    COLON = auto()          # :
    SEMICOLON = auto()      # ;
    NEWLINE = auto()        # \n (significant in some contexts)
    
    # Special
    EOF = auto()
    ERROR = auto()
    COMMENT = auto()        # For documentation extraction
    DOC_COMMENT = auto()    # /// style comments


# Keyword mapping
KEYWORDS: dict[str, TokenType] = {
    # Control flow
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "elif": TokenType.ELIF,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "return": TokenType.RETURN,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    
    # Definitions
    "fn": TokenType.FN,
    "let": TokenType.LET,
    "mut": TokenType.MUT,
    "type": TokenType.TYPE,
    "struct": TokenType.STRUCT,
    "enum": TokenType.ENUM,
    "impl": TokenType.IMPL,
    "trait": TokenType.TRAIT,
    
    # Delta-specific
    "param": TokenType.PARAM,
    "obs": TokenType.OBS,
    "const": TokenType.CONST,
    "learn": TokenType.LEARN,
    "constraint": TokenType.CONSTRAINT,
    "require": TokenType.REQUIRE,
    "prefer": TokenType.PREFER,
    "temperature": TokenType.TEMPERATURE,
    "temp": TokenType.TEMPERATURE,
    "non_diff": TokenType.NON_DIFF,
    "rand": TokenType.RAND,
    "observe": TokenType.OBSERVE,
    "where": TokenType.WHERE,
    
    # Mode
    "train": TokenType.TRAIN,
    "infer": TokenType.INFER,
    "analyze": TokenType.ANALYZE,
    
    # Types
    "Tensor": TokenType.TENSOR,
    "Scalar": TokenType.SCALAR,
    "Bool": TokenType.BOOL,
    "Int": TokenType.INT,
    "Float": TokenType.FLOAT,
    "Shape": TokenType.SHAPE,
    
    # Other
    "import": TokenType.IMPORT,
    "from": TokenType.FROM,
    "as": TokenType.AS,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "none": TokenType.NONE,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
}


@dataclass(frozen=True, slots=True)
class Token:
    """
    A single token from lexical analysis.
    
    Tokens are immutable and carry their source location
    for accurate error reporting.
    """
    type: TokenType
    value: Any
    location: SourceLocation
    
    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.type.name}({self.value!r})"
        return self.type.name
    
    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.location})"
    
    @property
    def is_operator(self) -> bool:
        """Check if this is an operator token."""
        return self.type in {
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.PERCENT, TokenType.DOUBLE_STAR, TokenType.AT,
            TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.LE,
            TokenType.GT, TokenType.GE, TokenType.AND, TokenType.OR,
            TokenType.NOT, TokenType.PIPE, TokenType.AMPERSAND, TokenType.CARET,
        }
    
    @property
    def is_assignment(self) -> bool:
        """Check if this is an assignment operator."""
        return self.type in {
            TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
            TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN,
        }
    
    @property
    def is_literal(self) -> bool:
        """Check if this is a literal token."""
        return self.type in {
            TokenType.INT_LITERAL, TokenType.FLOAT_LITERAL,
            TokenType.STRING_LITERAL, TokenType.BOOL_LITERAL,
            TokenType.TRUE, TokenType.FALSE, TokenType.NONE,
        }
    
    @property
    def is_keyword(self) -> bool:
        """Check if this is a keyword."""
        return self.type in KEYWORDS.values()
    
    def matches(self, *types: TokenType) -> bool:
        """Check if token matches any of the given types."""
        return self.type in types


# Operator precedence levels (higher = tighter binding)
PRECEDENCE: dict[TokenType, int] = {
    TokenType.OR: 1,
    TokenType.AND: 2,
    TokenType.PIPE: 3,
    TokenType.CARET: 4,
    TokenType.AMPERSAND: 5,
    TokenType.EQ: 6,
    TokenType.NE: 6,
    TokenType.LT: 7,
    TokenType.LE: 7,
    TokenType.GT: 7,
    TokenType.GE: 7,
    TokenType.PLUS: 8,
    TokenType.MINUS: 8,
    TokenType.STAR: 9,
    TokenType.SLASH: 9,
    TokenType.PERCENT: 9,
    TokenType.AT: 9,
    TokenType.DOUBLE_STAR: 10,  # Right associative
    TokenType.NOT: 11,
    TokenType.TILDE: 11,
}


# Right-associative operators
RIGHT_ASSOCIATIVE: set[TokenType] = {
    TokenType.DOUBLE_STAR,
    TokenType.ASSIGN,
    TokenType.PLUS_ASSIGN,
    TokenType.MINUS_ASSIGN,
    TokenType.STAR_ASSIGN,
    TokenType.SLASH_ASSIGN,
}
