"""
Delta frontend: lexer, parser, and AST definitions.

The frontend parses Delta source code into a typed AST with
source locations preserved throughout.
"""

from delta.frontend.tokens import Token, TokenType
from delta.frontend.lexer import Lexer
from delta.frontend.ast import (
    ASTNode, Module, Statement, Expression,
    FunctionDef, LearnBlock, ConstraintStmt,
    IfExpr, BinaryOp, UnaryOp, Call, Tensor, Param, Obs,
)
from delta.frontend.parser import Parser
from delta.frontend.desugar import Desugarer

__all__ = [
    "Token", "TokenType",
    "Lexer",
    "ASTNode", "Module", "Statement", "Expression",
    "FunctionDef", "LearnBlock", "ConstraintStmt",
    "IfExpr", "BinaryOp", "UnaryOp", "Call", "Tensor", "Param", "Obs",
    "Parser",
    "Desugarer",
]
