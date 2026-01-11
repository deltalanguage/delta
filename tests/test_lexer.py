"""
Tests for the Delta lexer.
"""

import pytest
from delta.frontend.lexer import Lexer
from delta.frontend.tokens import TokenType
from delta.source import SourceFile


class TestLexerBasics:
    """Basic lexer functionality tests."""
    
    def test_empty_source(self):
        """Lexing empty source produces only EOF."""
        source = SourceFile("<test>", "")
        lexer = Lexer(source)
        tokens = list(lexer)
        
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_whitespace_only(self):
        """Whitespace-only source produces NEWLINEs and EOF."""
        source = SourceFile("<test>", "   \n\t  \n  ")
        lexer = Lexer(source)
        tokens = list(lexer)
        
        # Lexer emits NEWLINE tokens for line breaks
        assert tokens[-1].type == TokenType.EOF
        # Filter out NEWLINEs for count
        non_newline = [t for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert len(non_newline) == 0
    
    def test_comment(self):
        """Comments are tokenized (using // syntax)."""
        source = SourceFile("<test>", "// this is a comment\n42")
        lexer = Lexer(source)
        tokens = list(lexer)
        
        # Filter out COMMENT and NEWLINE tokens
        significant = [t for t in tokens if t.type not in (TokenType.COMMENT, TokenType.NEWLINE, TokenType.EOF)]
        assert len(significant) == 1
        assert significant[0].type == TokenType.INT_LITERAL
        assert significant[0].value == 42


class TestLexerKeywords:
    """Keyword recognition tests."""
    
    @pytest.mark.parametrize("keyword,expected", [
        ("param", TokenType.PARAM),
        ("obs", TokenType.OBS),
        ("constraint", TokenType.CONSTRAINT),
        ("if", TokenType.IF),
        ("else", TokenType.ELSE),
        ("while", TokenType.WHILE),
        ("for", TokenType.FOR),
        ("in", TokenType.IN),
        ("fn", TokenType.FN),
        ("return", TokenType.RETURN),
        ("let", TokenType.LET),
        ("true", TokenType.TRUE),
        ("false", TokenType.FALSE),
        ("and", TokenType.AND),
        ("or", TokenType.OR),
        ("not", TokenType.NOT),
        ("train", TokenType.TRAIN),
        ("infer", TokenType.INFER),
        ("non_diff", TokenType.NON_DIFF),
    ])
    def test_keyword(self, keyword: str, expected: TokenType):
        """Keywords are correctly recognized."""
        source = SourceFile("<test>", keyword)
        lexer = Lexer(source)
        tokens = list(lexer)
        
        assert tokens[0].type == expected


class TestLexerLiterals:
    """Literal value tests."""
    
    def test_integer_literals(self):
        """Integer literals are parsed correctly."""
        source = SourceFile("<test>", "0 42 123456789")
        lexer = Lexer(source)
        tokens = list(lexer)
        
        assert tokens[0].type == TokenType.INT_LITERAL
        assert tokens[0].value == 0
        assert tokens[1].type == TokenType.INT_LITERAL
        assert tokens[1].value == 42
        assert tokens[2].type == TokenType.INT_LITERAL
        assert tokens[2].value == 123456789
    
    def test_float_literals(self):
        """Float literals are parsed correctly."""
        source = SourceFile("<test>", "0.0 3.14 1e-10 2.5e+3")
        lexer = Lexer(source)
        tokens = list(lexer)
        
        assert tokens[0].type == TokenType.FLOAT_LITERAL
        assert tokens[0].value == 0.0
        assert tokens[1].type == TokenType.FLOAT_LITERAL
        assert abs(tokens[1].value - 3.14) < 1e-10
        assert tokens[2].type == TokenType.FLOAT_LITERAL
        assert abs(tokens[2].value - 1e-10) < 1e-15
    
    def test_string_literals(self):
        """String literals are parsed correctly."""
        source = SourceFile("<test>", '"hello" "world" "with spaces"')
        lexer = Lexer(source)
        tokens = list(lexer)
        
        assert tokens[0].type == TokenType.STRING_LITERAL
        assert tokens[0].value == "hello"
        assert tokens[1].type == TokenType.STRING_LITERAL
        assert tokens[1].value == "world"
        assert tokens[2].type == TokenType.STRING_LITERAL
        assert tokens[2].value == "with spaces"


class TestLexerOperators:
    """Operator tokenization tests."""
    
    @pytest.mark.parametrize("op,expected", [
        ("+", TokenType.PLUS),
        ("-", TokenType.MINUS),
        ("*", TokenType.STAR),
        ("/", TokenType.SLASH),
        ("==", TokenType.EQ),
        ("!=", TokenType.NE),
        ("<", TokenType.LT),
        ("<=", TokenType.LE),
        (">", TokenType.GT),
        (">=", TokenType.GE),
        ("=", TokenType.ASSIGN),
        ("(", TokenType.LPAREN),
        (")", TokenType.RPAREN),
        ("[", TokenType.LBRACKET),
        ("]", TokenType.RBRACKET),
        ("{", TokenType.LBRACE),
        ("}", TokenType.RBRACE),
        (",", TokenType.COMMA),
        (":", TokenType.COLON),
        (";", TokenType.SEMICOLON),
        (".", TokenType.DOT),
        ("->", TokenType.ARROW),
    ])
    def test_operator(self, op: str, expected: TokenType):
        """Operators are correctly recognized."""
        source = SourceFile("<test>", op)
        lexer = Lexer(source)
        tokens = list(lexer)
        
        assert tokens[0].type == expected


class TestLexerIdentifiers:
    """Identifier tokenization tests."""
    
    @pytest.mark.parametrize("ident", [
        "x", "foo", "bar_baz", "_private", "CamelCase", "with123numbers"
    ])
    def test_valid_identifiers(self, ident: str):
        """Valid identifiers are recognized."""
        source = SourceFile("<test>", ident)
        lexer = Lexer(source)
        tokens = list(lexer)
        
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == ident


class TestLexerPositions:
    """Source position tracking tests."""
    
    def test_line_tracking(self):
        """Line numbers are tracked correctly."""
        source = SourceFile("<test>", "a\nb\nc")
        lexer = Lexer(source)
        tokens = list(lexer)
        
        # Filter to identifiers only (skip NEWLINE tokens)
        idents = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert idents[0].location.line == 1
        assert idents[1].location.line == 2
        assert idents[2].location.line == 3
    
    def test_column_tracking(self):
        """Column numbers are tracked correctly."""
        source = SourceFile("<test>", "abc def ghi")
        lexer = Lexer(source)
        tokens = list(lexer)
        
        assert tokens[0].location.column == 1
        assert tokens[1].location.column == 5
        assert tokens[2].location.column == 9


class TestLexerErrors:
    """Lexer error handling tests."""
    
    def test_unterminated_string(self):
        """Unterminated string raises error."""
        source = SourceFile("<test>", '"unterminated')
        lexer = Lexer(source)
        
        with pytest.raises(Exception):  # Should raise LexError
            list(lexer)
    
    def test_invalid_character(self):
        """Invalid character raises error."""
        source = SourceFile("<test>", "valid $ invalid")
        lexer = Lexer(source)
        
        with pytest.raises(Exception):  # Should raise LexError
            list(lexer)


class TestLexerComplex:
    """Complex input tests."""
    
    def test_function_definition(self):
        """Function definition is tokenized correctly."""
        source = SourceFile("<test>", "fn foo(x: Tensor) -> Tensor { return x * 2; }")
        lexer = Lexer(source)
        tokens = list(lexer)
        
        # Tensor is now a keyword (TENSOR), not an identifier
        expected_types = [
            TokenType.FN, TokenType.IDENTIFIER, TokenType.LPAREN,
            TokenType.IDENTIFIER, TokenType.COLON, TokenType.TENSOR,
            TokenType.RPAREN, TokenType.ARROW, TokenType.TENSOR,
            TokenType.LBRACE, TokenType.RETURN, TokenType.IDENTIFIER,
            TokenType.STAR, TokenType.INT_LITERAL, TokenType.SEMICOLON,
            TokenType.RBRACE, TokenType.EOF
        ]
        
        assert [t.type for t in tokens] == expected_types
    
    def test_constraint_expression(self):
        """Constraint expression is tokenized correctly."""
        source = SourceFile("<test>", "constraint x > 0 weight 1.0")
        lexer = Lexer(source)
        tokens = list(lexer)
        
        expected_types = [
            TokenType.CONSTRAINT, TokenType.IDENTIFIER, TokenType.GT,
            TokenType.INT_LITERAL, TokenType.IDENTIFIER, TokenType.FLOAT_LITERAL,
            TokenType.EOF
        ]
        
        assert [t.type for t in tokens] == expected_types
