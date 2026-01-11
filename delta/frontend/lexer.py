"""
Lexer for Delta language.

Performs lexical analysis, converting source text into a stream of tokens.
Handles all Delta-specific tokens including temperature annotations,
constraint keywords, and mode markers.
"""

from __future__ import annotations
from typing import Iterator, Optional
import re

from delta.source import SourceFile, SourceLocation
from delta.frontend.tokens import Token, TokenType, KEYWORDS
from delta.errors import LexerError, ErrorCode


class Lexer:
    """
    Delta lexer implementation.
    
    Converts source text into tokens while tracking precise
    source locations for error reporting.
    """
    
    def __init__(self, source: SourceFile) -> None:
        self.source = source
        self.content = source.content
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []
        self._peeked: Optional[Token] = None
    
    @classmethod
    def tokenize(cls, source: SourceFile) -> list[Token]:
        """Tokenize an entire source file."""
        lexer = cls(source)
        tokens = list(lexer)
        return tokens
    
    def __iter__(self) -> Iterator[Token]:
        """Iterate over all tokens in the source."""
        while True:
            token = self.next_token()
            yield token
            if token.type == TokenType.EOF:
                break
    
    @property
    def current_location(self) -> SourceLocation:
        """Get current source location."""
        return SourceLocation(
            file=self.source.path,
            line=self.line,
            column=self.column
        )
    
    def peek_char(self, offset: int = 0) -> str:
        """Peek at character at current position + offset."""
        pos = self.pos + offset
        if pos >= len(self.content):
            return '\0'
        return self.content[pos]
    
    def advance(self) -> str:
        """Advance position and return consumed character."""
        if self.pos >= len(self.content):
            return '\0'
        
        char = self.content[self.pos]
        self.pos += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        
        return char
    
    def skip_whitespace(self) -> None:
        """Skip whitespace (but not newlines in some contexts)."""
        while self.peek_char() in ' \t\r':
            self.advance()
    
    def skip_line(self) -> None:
        """Skip to end of line."""
        while self.peek_char() not in ('\n', '\0'):
            self.advance()
    
    def match(self, expected: str) -> bool:
        """Match and consume a character if it matches."""
        if self.peek_char() == expected:
            self.advance()
            return True
        return False
    
    def make_token(self, type: TokenType, value: object = None, 
                   start_loc: Optional[SourceLocation] = None) -> Token:
        """Create a token with location tracking."""
        loc = start_loc or self.current_location
        return Token(type=type, value=value, location=loc)
    
    def error(self, message: str, code: ErrorCode) -> LexerError:
        """Create a lexer error at current position."""
        return LexerError(
            message=message,
            location=self.current_location,
            code=code,
            source=self.source
        )
    
    def next_token(self) -> Token:
        """Get the next token from the input."""
        self.skip_whitespace()
        
        if self.pos >= len(self.content):
            return self.make_token(TokenType.EOF)
        
        start_loc = self.current_location
        char = self.peek_char()
        
        # Newlines (significant for statement separation)
        if char == '\n':
            self.advance()
            return self.make_token(TokenType.NEWLINE, start_loc=start_loc)
        
        # Comments
        if char == '/' and self.peek_char(1) == '/':
            return self.lex_comment(start_loc)
        
        # Block comments
        if char == '/' and self.peek_char(1) == '*':
            return self.lex_block_comment(start_loc)
        
        # String literals
        if char == '"':
            return self.lex_string(start_loc)
        
        # Numbers
        if char.isdigit() or (char == '.' and self.peek_char(1).isdigit()):
            return self.lex_number(start_loc)
        
        # Identifiers and keywords
        if char.isalpha() or char == '_':
            return self.lex_identifier(start_loc)
        
        # Operators and delimiters
        return self.lex_operator(start_loc)
    
    def lex_comment(self, start_loc: SourceLocation) -> Token:
        """Lex a line comment."""
        self.advance()  # First /
        self.advance()  # Second /
        
        # Check for doc comment ///
        is_doc = self.peek_char() == '/'
        if is_doc:
            self.advance()
        
        start = self.pos
        self.skip_line()
        content = self.content[start:self.pos].strip()
        
        token_type = TokenType.DOC_COMMENT if is_doc else TokenType.COMMENT
        return self.make_token(token_type, content, start_loc)
    
    def lex_block_comment(self, start_loc: SourceLocation) -> Token:
        """Lex a block comment /* */."""
        self.advance()  # /
        self.advance()  # *
        
        start = self.pos
        depth = 1
        
        while depth > 0 and self.pos < len(self.content):
            if self.peek_char() == '/' and self.peek_char(1) == '*':
                depth += 1
                self.advance()
                self.advance()
            elif self.peek_char() == '*' and self.peek_char(1) == '/':
                depth -= 1
                self.advance()
                self.advance()
            else:
                self.advance()
        
        if depth > 0:
            raise self.error(
                "Unterminated block comment",
                ErrorCode.E004_UNTERMINATED_COMMENT
            )
        
        content = self.content[start:self.pos - 2]
        return self.make_token(TokenType.COMMENT, content, start_loc)
    
    def lex_string(self, start_loc: SourceLocation) -> Token:
        """Lex a string literal with escape sequences."""
        self.advance()  # Opening quote
        
        chars: list[str] = []
        while self.peek_char() != '"':
            if self.peek_char() == '\0':
                raise self.error(
                    "Unterminated string literal",
                    ErrorCode.E002_UNTERMINATED_STRING
                )
            
            if self.peek_char() == '\\':
                self.advance()
                escaped = self.advance()
                escape_map = {
                    'n': '\n', 't': '\t', 'r': '\r',
                    '\\': '\\', '"': '"', '0': '\0',
                }
                if escaped in escape_map:
                    chars.append(escape_map[escaped])
                elif escaped == 'x':
                    # Hex escape \xNN
                    hex_chars = self.advance() + self.advance()
                    try:
                        chars.append(chr(int(hex_chars, 16)))
                    except ValueError:
                        raise self.error(
                            f"Invalid hex escape: \\x{hex_chars}",
                            ErrorCode.E002_UNTERMINATED_STRING
                        )
                else:
                    chars.append(escaped)
            else:
                chars.append(self.advance())
        
        self.advance()  # Closing quote
        return self.make_token(TokenType.STRING_LITERAL, ''.join(chars), start_loc)
    
    def lex_number(self, start_loc: SourceLocation) -> Token:
        """Lex an integer or float literal."""
        start = self.pos
        is_float = False
        
        # Handle different bases
        if self.peek_char() == '0':
            next_char = self.peek_char(1).lower()
            if next_char == 'x':
                # Hexadecimal
                self.advance()
                self.advance()
                while self.peek_char() in '0123456789abcdefABCDEF_':
                    self.advance()
                text = self.content[start:self.pos].replace('_', '')
                try:
                    value = int(text, 16)
                except ValueError:
                    raise self.error(f"Invalid hex literal: {text}", ErrorCode.E003_INVALID_NUMBER)
                return self.make_token(TokenType.INT_LITERAL, value, start_loc)
            elif next_char == 'b':
                # Binary
                self.advance()
                self.advance()
                while self.peek_char() in '01_':
                    self.advance()
                text = self.content[start:self.pos].replace('_', '')
                try:
                    value = int(text, 2)
                except ValueError:
                    raise self.error(f"Invalid binary literal: {text}", ErrorCode.E003_INVALID_NUMBER)
                return self.make_token(TokenType.INT_LITERAL, value, start_loc)
            elif next_char == 'o':
                # Octal
                self.advance()
                self.advance()
                while self.peek_char() in '01234567_':
                    self.advance()
                text = self.content[start:self.pos].replace('_', '')
                try:
                    value = int(text, 8)
                except ValueError:
                    raise self.error(f"Invalid octal literal: {text}", ErrorCode.E003_INVALID_NUMBER)
                return self.make_token(TokenType.INT_LITERAL, value, start_loc)
        
        # Decimal integer part
        while self.peek_char().isdigit() or self.peek_char() == '_':
            self.advance()
        
        # Decimal point
        if self.peek_char() == '.' and self.peek_char(1).isdigit():
            is_float = True
            self.advance()
            while self.peek_char().isdigit() or self.peek_char() == '_':
                self.advance()
        
        # Exponent
        if self.peek_char().lower() == 'e':
            is_float = True
            self.advance()
            if self.peek_char() in '+-':
                self.advance()
            if not self.peek_char().isdigit():
                raise self.error(
                    "Invalid number: expected exponent digits",
                    ErrorCode.E003_INVALID_NUMBER
                )
            while self.peek_char().isdigit() or self.peek_char() == '_':
                self.advance()
        
        text = self.content[start:self.pos].replace('_', '')
        
        try:
            if is_float:
                value = float(text)
                return self.make_token(TokenType.FLOAT_LITERAL, value, start_loc)
            else:
                value = int(text)
                return self.make_token(TokenType.INT_LITERAL, value, start_loc)
        except ValueError:
            raise self.error(f"Invalid number: {text}", ErrorCode.E003_INVALID_NUMBER)
    
    def lex_identifier(self, start_loc: SourceLocation) -> Token:
        """Lex an identifier or keyword."""
        start = self.pos
        
        while self.peek_char().isalnum() or self.peek_char() == '_':
            self.advance()
        
        text = self.content[start:self.pos]
        
        # Check for keywords
        if text in KEYWORDS:
            return self.make_token(KEYWORDS[text], text, start_loc)
        
        # Check for boolean literals
        if text == 'true':
            return self.make_token(TokenType.TRUE, True, start_loc)
        if text == 'false':
            return self.make_token(TokenType.FALSE, False, start_loc)
        
        return self.make_token(TokenType.IDENTIFIER, text, start_loc)
    
    def lex_operator(self, start_loc: SourceLocation) -> Token:
        """Lex an operator or delimiter."""
        char = self.advance()
        
        # Two-character operators first
        next_char = self.peek_char()
        
        two_char = char + next_char
        two_char_ops = {
            '==': TokenType.EQ,
            '!=': TokenType.NE,
            '<=': TokenType.LE,
            '>=': TokenType.GE,
            '->': TokenType.ARROW,
            '=>': TokenType.FAT_ARROW,
            '::': TokenType.DOUBLE_COLON,
            '**': TokenType.DOUBLE_STAR,
            '..': TokenType.DOUBLE_DOT,
            '+=': TokenType.PLUS_ASSIGN,
            '-=': TokenType.MINUS_ASSIGN,
            '*=': TokenType.STAR_ASSIGN,
            '/=': TokenType.SLASH_ASSIGN,
        }
        
        if two_char in two_char_ops:
            self.advance()
            return self.make_token(two_char_ops[two_char], two_char, start_loc)
        
        # Single-character operators
        single_char_ops = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.STAR,
            '/': TokenType.SLASH,
            '%': TokenType.PERCENT,
            '@': TokenType.AT,
            '<': TokenType.LT,
            '>': TokenType.GT,
            '=': TokenType.ASSIGN,
            '|': TokenType.PIPE,
            '&': TokenType.AMPERSAND,
            '^': TokenType.CARET,
            '~': TokenType.TILDE,
            '.': TokenType.DOT,
            '?': TokenType.QUESTION,
            '!': TokenType.BANG,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            ',': TokenType.COMMA,
            ':': TokenType.COLON,
            ';': TokenType.SEMICOLON,
        }
        
        if char in single_char_ops:
            return self.make_token(single_char_ops[char], char, start_loc)
        
        # Unknown character
        raise self.error(
            f"Unexpected character: {char!r}",
            ErrorCode.E001_UNEXPECTED_CHARACTER
        )
    
    def peek_token(self) -> Token:
        """Peek at next token without consuming it."""
        if self._peeked is None:
            self._peeked = self.next_token()
        return self._peeked
    
    def consume_token(self) -> Token:
        """Consume and return the next token."""
        if self._peeked is not None:
            token = self._peeked
            self._peeked = None
            return token
        return self.next_token()
