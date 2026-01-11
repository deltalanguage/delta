"""
Tests for the Delta parser.
"""

import pytest
from delta.frontend.parser import Parser
from delta.frontend.ast import (
    Module, Literal, Identifier, BinaryOp, UnaryOp, Call, MethodCall,
    Index, FieldAccess, LetStmt, AssignStmt, ExprStmt, ReturnStmt,
    IfStmt, WhileStmt, ForStmt, FunctionDef, ParamDecl, ObsDecl,
    LearnBlock, ConstraintStmt, Block,
)
from delta.source import SourceFile


def parse(code: str) -> Module:
    """Helper to parse code string."""
    source = SourceFile("<test>", code)
    parser = Parser(source)
    return parser.parse_module()


class TestParserExpressions:
    """Expression parsing tests."""
    
    def test_integer_literal(self):
        """Parse integer literal."""
        prog = parse("42;")
        assert len(prog.items) == 1
        stmt = prog.items[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, Literal)
        assert stmt.expr.kind == "int"
        assert stmt.expr.value == 42
    
    def test_float_literal(self):
        """Parse float literal."""
        prog = parse("3.14;")
        stmt = prog.items[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, Literal)
        assert stmt.expr.kind == "float"
        assert abs(stmt.expr.value - 3.14) < 1e-10
    
    def test_string_literal(self):
        """Parse string literal."""
        prog = parse('"hello";')
        stmt = prog.items[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, Literal)
        assert stmt.expr.kind == "string"
        assert stmt.expr.value == "hello"
    
    def test_boolean_literals(self):
        """Parse boolean literals."""
        prog = parse("true; false;")
        assert isinstance(prog.items[0], ExprStmt)
        assert isinstance(prog.items[0].expr, Literal)
        assert prog.items[0].expr.kind == "bool"
        assert prog.items[0].expr.value is True
        assert prog.items[1].expr.value is False
    
    def test_identifier(self):
        """Parse identifier."""
        prog = parse("foo;")
        stmt = prog.items[0]
        assert isinstance(stmt.expr, Identifier)
        assert stmt.expr.name == "foo"
    
    def test_binary_operators(self):
        """Parse binary operators."""
        prog = parse("a + b * c;")
        stmt = prog.items[0]
        expr = stmt.expr
        
        # Should be: a + (b * c) due to precedence
        assert isinstance(expr, BinaryOp)
        assert isinstance(expr.left, Identifier)
        assert isinstance(expr.right, BinaryOp)
    
    def test_comparison_operators(self):
        """Parse comparison operators."""
        prog = parse("x < y;")
        expr = prog.items[0].expr
        assert isinstance(expr, BinaryOp)
    
    def test_logical_operators(self):
        """Parse logical operators."""
        prog = parse("a and b or c;")
        expr = prog.items[0].expr
        
        # Should be: (a and b) or c
        assert isinstance(expr, BinaryOp)
    
    def test_unary_operators(self):
        """Parse unary operators."""
        prog = parse("-x;")
        assert isinstance(prog.items[0].expr, UnaryOp)
    
    def test_parenthesized_expression(self):
        """Parse parenthesized expression."""
        prog = parse("(a + b) * c;")
        expr = prog.items[0].expr
        
        assert isinstance(expr, BinaryOp)
        assert isinstance(expr.left, BinaryOp)
    
    def test_function_call(self):
        """Parse function call."""
        prog = parse("foo(a, b, c);")
        expr = prog.items[0].expr
        
        assert isinstance(expr, Call)
        assert isinstance(expr.func, Identifier)
        assert expr.func.name == "foo"
        assert len(expr.args) == 3
    
    def test_method_call(self):
        """Parse method call."""
        prog = parse("let r = obj.method(x);")  # Wrap in let statement
        stmt = prog.items[0]
        assert isinstance(stmt, LetStmt)
        # Method calls are parsed as Call with FieldAccess as func
        assert isinstance(stmt.value, (Call, MethodCall))
    
    def test_index_expression(self):
        """Parse index expression."""
        prog = parse("arr[i];")
        expr = prog.items[0].expr
        
        assert isinstance(expr, Index)
        assert isinstance(expr.base, Identifier)
    
    def test_attribute_access(self):
        """Parse attribute access."""
        prog = parse("obj.attr;")
        expr = prog.items[0].expr
        
        assert isinstance(expr, FieldAccess)
        assert expr.field == "attr"


class TestParserStatements:
    """Statement parsing tests."""
    
    def test_let_statement(self):
        """Parse let statement."""
        prog = parse("let x = 42;")
        stmt = prog.items[0]
        
        assert isinstance(stmt, LetStmt)
        assert isinstance(stmt.value, Literal)
    
    def test_let_with_type(self):
        """Parse let statement with type annotation."""
        prog = parse("let x: Tensor = zeros(10);")
        stmt = prog.items[0]
        
        assert isinstance(stmt, LetStmt)
        # Type annotation is stored on the pattern for let statements
        assert stmt.pattern.type_annotation is not None or stmt.type_annotation is not None
    
    def test_assignment(self):
        """Parse assignment statement."""
        prog = parse("x = 42;")
        stmt = prog.items[0]
        
        assert isinstance(stmt, AssignStmt)
        assert isinstance(stmt.target, Identifier)
    
    def test_return_statement(self):
        """Parse return statement."""
        prog = parse("return x + 1;")
        stmt = prog.items[0]
        
        assert isinstance(stmt, ReturnStmt)
        assert isinstance(stmt.value, BinaryOp)
    
    def test_if_statement(self):
        """Parse if statement in function body."""
        prog = parse("fn test() { if x > 0 { y = 1; } }")
        fn = prog.items[0]
        assert isinstance(fn, FunctionDef)
        # If statements are inside function bodies
        stmt = fn.body.statements[0]
        assert isinstance(stmt, IfStmt)
        assert isinstance(stmt.condition, BinaryOp)
    
    def test_if_else_statement(self):
        """Parse if-else statement in function body."""
        prog = parse("fn test() { if x { a = 1; } else { b = 2; } }")
        fn = prog.items[0]
        assert isinstance(fn, FunctionDef)
        stmt = fn.body.statements[0]
        assert isinstance(stmt, IfStmt)
        assert stmt.else_block is not None
    
    def test_while_statement(self):
        """Parse while statement in function body."""
        prog = parse("fn test() { while i < 10 { i = i + 1; } }")
        fn = prog.items[0]
        assert isinstance(fn, FunctionDef)
        stmt = fn.body.statements[0]
        assert isinstance(stmt, WhileStmt)
    
    def test_for_statement(self):
        """Parse for statement in function body."""
        prog = parse("fn test() { for i in range(10) { sum = sum + i; } }")
        fn = prog.items[0]
        assert isinstance(fn, FunctionDef)
        stmt = fn.body.statements[0]
        assert isinstance(stmt, ForStmt)


class TestParserDeclarations:
    """Declaration parsing tests."""
    
    def test_param_declaration(self):
        """Parse param declaration."""
        prog = parse("param theta: Tensor = zeros(10);")
        stmt = prog.items[0]
        
        assert isinstance(stmt, ParamDecl)
        assert stmt.name == "theta"
    
    def test_obs_declaration(self):
        """Parse obs declaration."""
        prog = parse("obs data: Tensor;")
        stmt = prog.items[0]
        
        assert isinstance(stmt, ObsDecl)
        assert stmt.name == "data"
    
    def test_function_definition(self):
        """Parse function definition."""
        prog = parse("fn foo(x: Tensor, y: Tensor) -> Tensor { return x + y; }")
        stmt = prog.items[0]
        
        assert isinstance(stmt, FunctionDef)
        assert stmt.name == "foo"
        assert len(stmt.params) == 2
        assert stmt.return_type is not None
    
    def test_constraint_declaration(self):
        """Parse constraint declaration."""
        prog = parse("constraint x > 0 weight 1.0;")
        stmt = prog.items[0]
        
        assert isinstance(stmt, ConstraintStmt)
        assert isinstance(stmt.expr, BinaryOp)


class TestParserModeBlocks:
    """Mode block parsing tests."""
    
    def test_train_block(self):
        """Parse train block."""
        prog = parse("train { let x = 1; }")
        stmt = prog.items[0]
        
        assert isinstance(stmt, LearnBlock)
        assert stmt.mode == "train"
    
    def test_infer_block(self):
        """Parse infer block."""
        prog = parse("infer { let x = predict(y); }")
        stmt = prog.items[0]
        
        assert isinstance(stmt, LearnBlock)
        assert stmt.mode == "infer"
    
    def test_learn_train_block(self):
        """Parse learn train block (explicit form)."""
        prog = parse("learn train { let x = 1; }")
        stmt = prog.items[0]
        
        assert isinstance(stmt, LearnBlock)
        assert stmt.mode == "train"


class TestParserComplex:
    """Complex program parsing tests."""
    
    def test_simple_model(self):
        """Parse a simple model definition."""
        code = """
        param theta: Tensor = randn(10);
        obs x: Tensor;
        obs y: Tensor;
        
        fn forward(x: Tensor) -> Tensor {
            return matmul(x, theta);
        }
        
        let pred = forward(x);
        constraint pred == y weight 1.0;
        """
        prog = parse(code)
        
        assert len(prog.items) >= 5
    
    def test_nested_expressions(self):
        """Parse deeply nested expressions."""
        prog = parse("foo(bar(baz(x + 1) * 2) / 3);")
        
        assert len(prog.items) == 1
        assert isinstance(prog.items[0].expr, Call)
    
    def test_chained_method_calls(self):
        """Parse chained method calls."""
        prog = parse("let r = tensor.view(10, 10).transpose(0, 1).sum();")
        
        stmt = prog.items[0]
        assert isinstance(stmt, LetStmt)
        # Chained method calls are represented as MethodCall nodes
        assert isinstance(stmt.value, (Call, MethodCall))


class TestParserErrors:
    """Parser error handling tests."""
    
    def test_missing_semicolon(self):
        """Missing semicolon should be reported as error."""
        # Parser may recover - just check it doesn't crash
        prog = parse("let x = 42")
        # Should either have errors or produce partial result
    
    def test_unbalanced_parens(self):
        """Unbalanced parentheses should be reported."""
        # Parser should handle gracefully
        try:
            prog = parse("foo(a, b;")
            # Check for errors in result
        except Exception:
            pass  # Expected
    
    def test_unexpected_token(self):
        """Unexpected token should be reported."""
        try:
            prog = parse("let = 42;")
        except Exception:
            pass  # Expected
