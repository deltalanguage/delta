"""
Recursive descent parser for Delta.

Parses token streams into AST nodes, handling all Delta syntax
including differentiable constructs, constraints, and role annotations.
"""

from __future__ import annotations
from typing import Optional, Callable
from dataclasses import dataclass

from delta.source import SourceFile, SourceLocation
from delta.frontend.tokens import Token, TokenType, PRECEDENCE, RIGHT_ASSOCIATIVE
from delta.frontend.lexer import Lexer
from delta.frontend.ast import (
    ASTNode, Module, Statement, Expression,
    FunctionDef, Parameter, StructDef, StructField,
    LetStmt, AssignStmt, ExprStmt, ReturnStmt,
    IfStmt, WhileStmt, ForStmt, LearnBlock, ConstraintStmt,
    ParamDecl, ObsDecl, ImportStmt,
    Block, IfExpr, BinaryOp, UnaryOp, Call, MethodCall,
    Index, FieldAccess, Tensor, Param, Obs, Identifier, Literal,
    Lambda, RandExpr, ObserveExpr, NonDiffBlock,
    BinaryOperator, UnaryOperator, ConstraintKind,
    TypeAnnotation, SimpleType, TensorType, FunctionType, GenericType,
    Pattern, IdentifierPattern, TuplePattern,
)
from delta.errors import ParseError, ErrorCode, ErrorCollector


# Operator mappings
BINARY_OP_MAP: dict[TokenType, BinaryOperator] = {
    TokenType.PLUS: BinaryOperator.ADD,
    TokenType.MINUS: BinaryOperator.SUB,
    TokenType.STAR: BinaryOperator.MUL,
    TokenType.SLASH: BinaryOperator.DIV,
    TokenType.PERCENT: BinaryOperator.MOD,
    TokenType.DOUBLE_STAR: BinaryOperator.POW,
    TokenType.AT: BinaryOperator.MATMUL,
    TokenType.EQ: BinaryOperator.EQ,
    TokenType.NE: BinaryOperator.NE,
    TokenType.LT: BinaryOperator.LT,
    TokenType.LE: BinaryOperator.LE,
    TokenType.GT: BinaryOperator.GT,
    TokenType.GE: BinaryOperator.GE,
    TokenType.AND: BinaryOperator.AND,
    TokenType.OR: BinaryOperator.OR,
    TokenType.AMPERSAND: BinaryOperator.BIT_AND,
    TokenType.PIPE: BinaryOperator.BIT_OR,
    TokenType.CARET: BinaryOperator.BIT_XOR,
    TokenType.TILDE: BinaryOperator.OBSERVE,
}

UNARY_OP_MAP: dict[TokenType, UnaryOperator] = {
    TokenType.MINUS: UnaryOperator.NEG,
    TokenType.NOT: UnaryOperator.NOT,
    TokenType.BANG: UnaryOperator.NOT,
    TokenType.TILDE: UnaryOperator.BIT_NOT,
}

COMPOUND_ASSIGN_MAP: dict[TokenType, BinaryOperator] = {
    TokenType.PLUS_ASSIGN: BinaryOperator.ADD,
    TokenType.MINUS_ASSIGN: BinaryOperator.SUB,
    TokenType.STAR_ASSIGN: BinaryOperator.MUL,
    TokenType.SLASH_ASSIGN: BinaryOperator.DIV,
}


class Parser:
    """
    Recursive descent parser for Delta.
    
    Produces an AST from a token stream with full source location
    tracking for error reporting.
    """
    
    def __init__(self, source: SourceFile) -> None:
        self.source = source
        self.lexer = Lexer(source)
        self.tokens: list[Token] = []
        self.pos = 0
        self.errors = ErrorCollector()
        self._tokenize()
    
    def _tokenize(self) -> None:
        """Pre-tokenize the entire source."""
        for token in self.lexer:
            # Skip comments for parsing (but could preserve for docs)
            if token.type not in (TokenType.COMMENT, TokenType.DOC_COMMENT):
                self.tokens.append(token)
    
    @classmethod
    def parse(cls, source: SourceFile) -> Module:
        """Parse a source file into a Module AST."""
        parser = cls(source)
        return parser.parse_module()
    
    @classmethod
    def parse_string(cls, code: str, name: str = "<string>") -> Module:
        """Parse a string into a Module AST."""
        source = SourceFile.from_string(code, name)
        return cls.parse(source)
    
    def current(self) -> Token:
        """Get current token."""
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[self.pos]
    
    def peek(self, offset: int = 0) -> Token:
        """Peek at token at offset from current position."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[pos]
    
    def advance(self) -> Token:
        """Advance and return current token."""
        token = self.current()
        if token.type != TokenType.EOF:
            self.pos += 1
        return token
    
    def check(self, *types: TokenType) -> bool:
        """Check if current token matches any of the types."""
        return self.current().type in types
    
    def match(self, *types: TokenType) -> Optional[Token]:
        """Match and consume if current token matches."""
        if self.check(*types):
            return self.advance()
        return None
    
    def expect(self, type: TokenType, message: Optional[str] = None) -> Token:
        """Expect and consume a specific token type."""
        if self.check(type):
            return self.advance()
        
        msg = message or f"Expected {type.name}"
        raise self.error(msg, ErrorCode.E101_UNEXPECTED_TOKEN)
    
    def error(self, message: str, code: ErrorCode) -> ParseError:
        """Create a parse error at current position."""
        return ParseError(
            message=f"{message}, got {self.current().type.name}",
            location=self.current().location,
            code=code,
            source=self.source
        )
    
    def skip_newlines(self) -> None:
        """Skip newline tokens."""
        while self.match(TokenType.NEWLINE):
            pass
    
    def at_statement_end(self) -> bool:
        """Check if at end of statement."""
        return self.check(TokenType.NEWLINE, TokenType.SEMICOLON, 
                         TokenType.RBRACE, TokenType.EOF)
    
    def expect_statement_end(self) -> None:
        """Expect end of statement."""
        if not self.at_statement_end():
            raise self.error("Expected end of statement", ErrorCode.E101_UNEXPECTED_TOKEN)
        self.skip_newlines()
        while self.match(TokenType.SEMICOLON):
            self.skip_newlines()
    
    # Top-level parsing
    
    def parse_module(self) -> Module:
        """Parse a complete module."""
        start_loc = self.current().location
        items: list[Statement] = []
        
        self.skip_newlines()
        
        while not self.check(TokenType.EOF):
            try:
                item = self.parse_top_level_item()
                if item:
                    items.append(item)
            except ParseError as e:
                self.errors.add(e)
                self.synchronize()
            
            self.skip_newlines()
        
        return Module(
            location=start_loc,
            name=self.source.path,
            items=items
        )
    
    def parse_top_level_item(self) -> Optional[Statement]:
        """Parse a top-level item."""
        self.skip_newlines()
        
        if self.check(TokenType.FN):
            return self.parse_function_def()
        elif self.check(TokenType.STRUCT):
            return self.parse_struct_def()
        elif self.check(TokenType.LET):
            return self.parse_let_stmt()
        elif self.check(TokenType.PARAM):
            return self.parse_param_decl()
        elif self.check(TokenType.OBS):
            return self.parse_obs_decl()
        elif self.check(TokenType.IMPORT, TokenType.FROM):
            return self.parse_import_stmt()
        elif self.check(TokenType.LEARN):
            return self.parse_learn_block()
        elif self.check(TokenType.TRAIN):
            # Standalone train block: train { ... } -> learn train { ... }
            return self.parse_standalone_mode_block()
        elif self.check(TokenType.INFER):
            # Standalone infer block: infer { ... } -> learn infer { ... }
            return self.parse_standalone_mode_block()
        elif self.check(TokenType.CONSTRAINT):
            # constraint expr weight X; -> require expr weight X;
            return self.parse_constraint_stmt(ConstraintKind.REQUIRE)
        elif self.check(TokenType.REQUIRE):
            return self.parse_constraint_stmt(ConstraintKind.REQUIRE)
        elif self.check(TokenType.PREFER):
            return self.parse_constraint_stmt(ConstraintKind.PREFER)
        else:
            # Expression statement
            return self.parse_statement()
    
    def synchronize(self) -> None:
        """Synchronize after parse error."""
        self.advance()
        
        while not self.check(TokenType.EOF):
            if self.peek(-1).type in (TokenType.SEMICOLON, TokenType.NEWLINE):
                if self.check(TokenType.FN, TokenType.LET, TokenType.STRUCT,
                              TokenType.IF, TokenType.WHILE, TokenType.FOR,
                              TokenType.RETURN, TokenType.LEARN, TokenType.PARAM,
                              TokenType.OBS, TokenType.IMPORT):
                    return
            self.advance()
    
    # Declarations
    
    def parse_function_def(self) -> FunctionDef:
        """Parse a function definition."""
        start = self.expect(TokenType.FN)
        name = self.expect(TokenType.IDENTIFIER, "Expected function name")
        
        # Type parameters
        type_params: list[str] = []
        if self.match(TokenType.LT):
            type_params = self.parse_type_params()
        
        # Parameters
        self.expect(TokenType.LPAREN, "Expected '(' after function name")
        params = self.parse_parameters()
        self.expect(TokenType.RPAREN, "Expected ')' after parameters")
        
        # Return type
        return_type: Optional[TypeAnnotation] = None
        if self.match(TokenType.ARROW):
            return_type = self.parse_type()
        
        # Body
        self.skip_newlines()
        body = self.parse_block()
        
        return FunctionDef(
            location=start.location,
            name=name.value,
            params=params,
            return_type=return_type,
            body=body,
            type_params=type_params
        )
    
    def parse_type_params(self) -> list[str]:
        """Parse type parameters <T, U, ...>."""
        params: list[str] = []
        
        if not self.check(TokenType.GT):
            params.append(self.expect(TokenType.IDENTIFIER).value)
            while self.match(TokenType.COMMA):
                params.append(self.expect(TokenType.IDENTIFIER).value)
        
        self.expect(TokenType.GT, "Expected '>' after type parameters")
        return params
    
    def parse_parameters(self) -> list[Parameter]:
        """Parse function parameters."""
        params: list[Parameter] = []
        
        if not self.check(TokenType.RPAREN):
            params.append(self.parse_parameter())
            while self.match(TokenType.COMMA):
                if self.check(TokenType.RPAREN):
                    break
                params.append(self.parse_parameter())
        
        return params
    
    def parse_parameter(self) -> Parameter:
        """Parse a single parameter."""
        start = self.current()
        
        # Optional role annotation
        role: Optional[str] = None
        if self.match(TokenType.PARAM):
            role = "param"
        elif self.match(TokenType.OBS):
            role = "obs"
        
        name = self.expect(TokenType.IDENTIFIER, "Expected parameter name")
        
        # Type annotation
        type_ann: Optional[TypeAnnotation] = None
        if self.match(TokenType.COLON):
            type_ann = self.parse_type()
        
        # Default value
        default: Optional[Expression] = None
        if self.match(TokenType.ASSIGN):
            default = self.parse_expression()
        
        return Parameter(
            location=start.location,
            name=name.value,
            type_annotation=type_ann,
            default=default,
            role=role
        )
    
    def parse_struct_def(self) -> StructDef:
        """Parse a struct definition."""
        start = self.expect(TokenType.STRUCT)
        name = self.expect(TokenType.IDENTIFIER, "Expected struct name")
        
        # Type parameters
        type_params: list[str] = []
        if self.match(TokenType.LT):
            type_params = self.parse_type_params()
        
        self.skip_newlines()
        self.expect(TokenType.LBRACE, "Expected '{' after struct name")
        self.skip_newlines()
        
        # Fields
        fields: list[StructField] = []
        while not self.check(TokenType.RBRACE, TokenType.EOF):
            fields.append(self.parse_struct_field())
            if not self.match(TokenType.COMMA):
                self.skip_newlines()
            self.skip_newlines()
        
        self.expect(TokenType.RBRACE, "Expected '}'")
        
        return StructDef(
            location=start.location,
            name=name.value,
            fields=fields,
            type_params=type_params
        )
    
    def parse_struct_field(self) -> StructField:
        """Parse a struct field."""
        start = self.current()
        name = self.expect(TokenType.IDENTIFIER, "Expected field name")
        self.expect(TokenType.COLON, "Expected ':' after field name")
        type_ann = self.parse_type()
        
        default: Optional[Expression] = None
        if self.match(TokenType.ASSIGN):
            default = self.parse_expression()
        
        return StructField(
            location=start.location,
            name=name.value,
            type_annotation=type_ann,
            default=default
        )
    
    def parse_param_decl(self) -> ParamDecl:
        """Parse a param declaration."""
        start = self.expect(TokenType.PARAM)
        name = self.expect(TokenType.IDENTIFIER, "Expected parameter name")
        
        type_ann: Optional[TypeAnnotation] = None
        if self.match(TokenType.COLON):
            type_ann = self.parse_type()
        
        init: Optional[Expression] = None
        if self.match(TokenType.ASSIGN):
            init = self.parse_expression()
        
        self.expect_statement_end()
        
        return ParamDecl(
            location=start.location,
            name=name.value,
            type_annotation=type_ann,
            initializer=init
        )
    
    def parse_obs_decl(self) -> ObsDecl:
        """Parse an obs declaration."""
        start = self.expect(TokenType.OBS)
        name = self.expect(TokenType.IDENTIFIER, "Expected observation name")
        
        type_ann: Optional[TypeAnnotation] = None
        if self.match(TokenType.COLON):
            type_ann = self.parse_type()
        
        self.expect_statement_end()
        
        return ObsDecl(
            location=start.location,
            name=name.value,
            type_annotation=type_ann
        )
    
    def parse_import_stmt(self) -> ImportStmt:
        """Parse an import statement."""
        start = self.current()
        
        if self.match(TokenType.FROM):
            # from module import items
            path = self.parse_module_path()
            self.expect(TokenType.IMPORT, "Expected 'import' after module path")
            
            items: list[tuple[str, Optional[str]]] = []
            if self.match(TokenType.STAR):
                items.append(('*', None))
            else:
                items.append(self.parse_import_item())
                while self.match(TokenType.COMMA):
                    items.append(self.parse_import_item())
            
            self.expect_statement_end()
            return ImportStmt(location=start.location, module_path=path, items=items)
        else:
            # import module [as alias]
            self.expect(TokenType.IMPORT)
            path = self.parse_module_path()
            
            alias: Optional[str] = None
            if self.match(TokenType.AS):
                alias = self.expect(TokenType.IDENTIFIER).value
            
            self.expect_statement_end()
            return ImportStmt(location=start.location, module_path=path, alias=alias)
    
    def parse_module_path(self) -> list[str]:
        """Parse a module path like std.nn.linear."""
        path = [self.expect(TokenType.IDENTIFIER, "Expected module name").value]
        while self.match(TokenType.DOT):
            path.append(self.expect(TokenType.IDENTIFIER).value)
        return path
    
    def parse_import_item(self) -> tuple[str, Optional[str]]:
        """Parse an import item like 'foo' or 'foo as bar'."""
        name = self.expect(TokenType.IDENTIFIER, "Expected import name").value
        alias: Optional[str] = None
        if self.match(TokenType.AS):
            alias = self.expect(TokenType.IDENTIFIER).value
        return (name, alias)
    
    # Statements
    
    def parse_statement(self) -> Statement:
        """Parse a statement."""
        self.skip_newlines()
        
        if self.check(TokenType.LET):
            return self.parse_let_stmt()
        elif self.check(TokenType.RETURN):
            return self.parse_return_stmt()
        elif self.check(TokenType.IF):
            return self.parse_if_stmt()
        elif self.check(TokenType.WHILE):
            return self.parse_while_stmt()
        elif self.check(TokenType.FOR):
            return self.parse_for_stmt()
        elif self.check(TokenType.LEARN):
            return self.parse_learn_block()
        elif self.check(TokenType.REQUIRE):
            return self.parse_constraint_stmt(ConstraintKind.REQUIRE)
        elif self.check(TokenType.PREFER):
            return self.parse_constraint_stmt(ConstraintKind.PREFER)
        elif self.check(TokenType.CONSTRAINT):
            return self.parse_constraint_stmt(ConstraintKind.REQUIRE)
        elif self.check(TokenType.TRAIN):
            return self.parse_standalone_mode_block()
        elif self.check(TokenType.INFER):
            return self.parse_standalone_mode_block()
        elif self.check(TokenType.PARAM):
            return self.parse_param_decl()
        elif self.check(TokenType.OBS):
            return self.parse_obs_decl()
        else:
            return self.parse_expr_or_assign_stmt()
    
    def parse_let_stmt(self) -> LetStmt:
        """Parse a let statement."""
        start = self.expect(TokenType.LET)
        mutable = bool(self.match(TokenType.MUT))
        
        pattern = self.parse_pattern()
        
        type_ann: Optional[TypeAnnotation] = None
        if self.match(TokenType.COLON):
            type_ann = self.parse_type()
        
        self.expect(TokenType.ASSIGN, "Expected '=' in let binding")
        value = self.parse_expression()
        
        self.expect_statement_end()
        
        return LetStmt(
            location=start.location,
            pattern=pattern,
            type_annotation=type_ann,
            value=value,
            mutable=mutable
        )
    
    def parse_return_stmt(self) -> ReturnStmt:
        """Parse a return statement."""
        start = self.expect(TokenType.RETURN)
        
        value: Optional[Expression] = None
        if not self.at_statement_end():
            value = self.parse_expression()
        
        self.expect_statement_end()
        
        return ReturnStmt(location=start.location, value=value)
    
    def parse_if_stmt(self) -> IfStmt:
        """Parse an if statement."""
        start = self.expect(TokenType.IF)
        condition = self.parse_expression()
        
        # Optional temperature
        temperature: Optional[Expression] = None
        if self.match(TokenType.TEMPERATURE):
            temperature = self.parse_expression()
        
        self.skip_newlines()
        then_block = self.parse_block()
        
        else_block: Optional[Block | IfStmt] = None
        self.skip_newlines()
        if self.match(TokenType.ELSE):
            self.skip_newlines()
            if self.check(TokenType.IF):
                else_block = self.parse_if_stmt()
            else:
                else_block = self.parse_block()
        elif self.match(TokenType.ELIF):
            # Desugar elif into else { if ... }
            self.pos -= 1  # Back up
            self.tokens[self.pos] = Token(
                TokenType.IF, 
                self.tokens[self.pos].value,
                self.tokens[self.pos].location
            )
            else_block = self.parse_if_stmt()
        
        return IfStmt(
            location=start.location,
            condition=condition,
            then_block=then_block,
            else_block=else_block,
            temperature=temperature
        )
    
    def parse_while_stmt(self) -> WhileStmt:
        """Parse a while statement."""
        start = self.expect(TokenType.WHILE)
        condition = self.parse_expression()
        
        self.skip_newlines()
        body = self.parse_block()
        
        return WhileStmt(
            location=start.location,
            condition=condition,
            body=body
        )
    
    def parse_for_stmt(self) -> ForStmt:
        """Parse a for statement."""
        start = self.expect(TokenType.FOR)
        pattern = self.parse_pattern()
        self.expect(TokenType.IN, "Expected 'in' in for loop")
        iterable = self.parse_expression()
        
        self.skip_newlines()
        body = self.parse_block()
        
        return ForStmt(
            location=start.location,
            pattern=pattern,
            iterable=iterable,
            body=body
        )
    
    def parse_learn_block(self) -> LearnBlock:
        """Parse a learn block."""
        start = self.expect(TokenType.LEARN)
        
        # Optional mode
        mode = "train"
        if self.match(TokenType.TRAIN):
            mode = "train"
        elif self.match(TokenType.INFER):
            mode = "infer"
        elif self.match(TokenType.ANALYZE):
            mode = "analyze"
        
        self.skip_newlines()
        body = self.parse_block()
        
        # Optional 'with' clause
        optimizer: Optional[Expression] = None
        epochs: Optional[Expression] = None
        batch_size: Optional[Expression] = None
        
        self.skip_newlines()
        if self.match(TokenType.IDENTIFIER) and self.peek(-1).value == "with":
            while True:
                name = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.ASSIGN)
                value = self.parse_expression()
                
                if name == "optimizer":
                    optimizer = value
                elif name == "epochs":
                    epochs = value
                elif name == "batch_size":
                    batch_size = value
                
                if not self.match(TokenType.COMMA):
                    break
        
        return LearnBlock(
            location=start.location,
            body=body,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            mode=mode
        )
    
    def parse_standalone_mode_block(self) -> LearnBlock:
        """Parse a standalone train/infer block (syntactic sugar for learn train/infer)."""
        start = self.current()
        
        if self.match(TokenType.TRAIN):
            mode = "train"
        elif self.match(TokenType.INFER):
            mode = "infer"
        elif self.match(TokenType.ANALYZE):
            mode = "analyze"
        else:
            self.error("Expected 'train', 'infer', or 'analyze'")
            mode = "train"
        
        self.skip_newlines()
        body = self.parse_block()
        
        return LearnBlock(
            location=start.location,
            body=body,
            optimizer=None,
            epochs=None,
            batch_size=None,
            mode=mode
        )
    
    def parse_constraint_stmt(self, kind: ConstraintKind) -> ConstraintStmt:
        """Parse a constraint statement."""
        if self.check(TokenType.CONSTRAINT):
            start = self.expect(TokenType.CONSTRAINT)
        elif kind == ConstraintKind.REQUIRE:
            start = self.expect(TokenType.REQUIRE)
        else:
            start = self.expect(TokenType.PREFER)
        
        expr = self.parse_expression()
        
        weight: Optional[Expression] = None
        slack: Optional[Expression] = None
        
        # Parse optional weight and slack
        while self.check(TokenType.IDENTIFIER):
            if self.current().value == "weight":
                self.advance()
                weight = self.parse_expression()
            elif self.current().value == "slack":
                self.advance()
                slack = self.parse_expression()
            else:
                break
        
        self.expect_statement_end()
        
        return ConstraintStmt(
            location=start.location,
            kind=kind,
            expr=expr,
            weight=weight,
            slack=slack
        )
    
    def parse_expr_or_assign_stmt(self) -> Statement:
        """Parse an expression or assignment statement."""
        start = self.current()
        expr = self.parse_expression()
        
        # Check for assignment
        if self.check(TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
                      TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN):
            op_token = self.advance()
            value = self.parse_expression()
            
            op: Optional[BinaryOperator] = None
            if op_token.type in COMPOUND_ASSIGN_MAP:
                op = COMPOUND_ASSIGN_MAP[op_token.type]
            
            self.expect_statement_end()
            return AssignStmt(
                location=start.location,
                target=expr,
                value=value,
                op=op
            )
        
        self.expect_statement_end()
        return ExprStmt(location=start.location, expr=expr)
    
    # Blocks
    
    def parse_block(self) -> Block:
        """Parse a block { ... }."""
        start = self.expect(TokenType.LBRACE, "Expected '{'")
        self.skip_newlines()
        
        statements: list[Statement] = []
        result: Optional[Expression] = None
        
        while not self.check(TokenType.RBRACE, TokenType.EOF):
            # Try to parse a statement
            if self.check(TokenType.LET, TokenType.RETURN, TokenType.IF, 
                         TokenType.WHILE, TokenType.FOR, TokenType.REQUIRE,
                         TokenType.PREFER, TokenType.PARAM, TokenType.OBS,
                         TokenType.LEARN, TokenType.CONSTRAINT, TokenType.TRAIN,
                         TokenType.INFER):
                statements.append(self.parse_statement())
            else:
                # Could be expression statement, assignment, or trailing expression
                start = self.current()
                expr = self.parse_expression()
                
                # Check for assignment
                if self.check(TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
                              TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN):
                    op_token = self.advance()
                    value = self.parse_expression()
                    
                    op: Optional[BinaryOperator] = None
                    if op_token.type in COMPOUND_ASSIGN_MAP:
                        op = COMPOUND_ASSIGN_MAP[op_token.type]
                    
                    self.expect_statement_end()
                    statements.append(AssignStmt(
                        location=start.location,
                        target=expr,
                        value=value,
                        op=op
                    ))
                else:
                    # Expression statement or trailing expression
                    has_semicolon = self.match(TokenType.SEMICOLON)
                    
                    # Check what comes next (skip newlines to look ahead)
                    self.skip_newlines()
                    
                    if self.check(TokenType.RBRACE, TokenType.EOF):
                        # This is the last item in the block
                        if has_semicolon:
                            # Explicit semicolon = statement, not result
                            statements.append(ExprStmt(location=expr.location, expr=expr))
                        else:
                            # No semicolon = trailing expression (block result)
                            result = expr
                        break
                    else:
                        # Not at end of block - this is an expression statement
                        statements.append(ExprStmt(location=expr.location, expr=expr))
            
            self.skip_newlines()
        
        end = self.expect(TokenType.RBRACE, "Expected '}'")
        
        return Block(
            location=start.location.extend_to(end.location),
            statements=statements,
            result=result
        )
    
    # Patterns
    
    def parse_pattern(self) -> Pattern:
        """Parse a pattern for let bindings and function parameters."""
        start = self.current()
        
        if self.check(TokenType.LPAREN):
            # Tuple pattern
            self.advance()
            elements: list[Pattern] = []
            if not self.check(TokenType.RPAREN):
                elements.append(self.parse_pattern())
                while self.match(TokenType.COMMA):
                    if self.check(TokenType.RPAREN):
                        break
                    elements.append(self.parse_pattern())
            self.expect(TokenType.RPAREN)
            return TuplePattern(location=start.location, elements=elements)
        else:
            # Simple identifier pattern
            mutable = bool(self.match(TokenType.MUT))
            name = self.expect(TokenType.IDENTIFIER, "Expected pattern").value
            
            type_ann: Optional[TypeAnnotation] = None
            if self.match(TokenType.COLON):
                type_ann = self.parse_type()
            
            return IdentifierPattern(
                location=start.location,
                name=name,
                type_annotation=type_ann,
                mutable=mutable
            )
    
    # Types
    
    def parse_type(self) -> TypeAnnotation:
        """Parse a type annotation."""
        start = self.current()
        
        if self.check(TokenType.FN):
            return self.parse_function_type()
        elif self.check(TokenType.LPAREN):
            # Tuple or grouped type
            self.advance()
            if self.check(TokenType.RPAREN):
                self.advance()
                return SimpleType(location=start.location, name="Unit")
            
            first = self.parse_type()
            if self.match(TokenType.COMMA):
                # Tuple type
                types = [first]
                types.append(self.parse_type())
                while self.match(TokenType.COMMA):
                    types.append(self.parse_type())
                self.expect(TokenType.RPAREN)
                return GenericType(
                    location=start.location,
                    name="Tuple",
                    type_args=types
                )
            self.expect(TokenType.RPAREN)
            return first
        else:
            return self.parse_simple_type()
    
    def parse_simple_type(self) -> TypeAnnotation:
        """Parse a simple or generic type."""
        start = self.current()
        
        if self.check(TokenType.TENSOR):
            self.advance()
            if self.match(TokenType.LBRACKET):
                # Tensor with element type and optional shape
                element_type = self.parse_type()
                shape: list[Expression] = None
                if self.match(TokenType.COMMA):
                    # Shape dimensions follow
                    shape = []
                if not self.check(TokenType.RBRACKET):
                    shape.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        shape.append(self.parse_expression())
                self.expect(TokenType.RBRACKET)
                return TensorType(
                    location=start.location,
                    element_type=element_type,
                    shape=shape
                )
            return TensorType(
                location=start.location,
                element_type=SimpleType(location=start.location, name="Float"),
                shape=None
            )
        
        # Handle primitive type keywords
        if self.check(TokenType.FLOAT):
            self.advance()
            return SimpleType(location=start.location, name="Float")
        elif self.check(TokenType.INT):
            self.advance()
            return SimpleType(location=start.location, name="Int")
        elif self.check(TokenType.BOOL):
            self.advance()
            return SimpleType(location=start.location, name="Bool")
        elif self.check(TokenType.STRING):
            self.advance()
            return SimpleType(location=start.location, name="String")
        
        name = self.expect(TokenType.IDENTIFIER, "Expected type name")
        
        # Check for type arguments
        if self.match(TokenType.LBRACKET):
            type_args: list[TypeAnnotation] = []
            if not self.check(TokenType.RBRACKET):
                type_args.append(self.parse_type())
                while self.match(TokenType.COMMA):
                    type_args.append(self.parse_type())
            self.expect(TokenType.RBRACKET)
            return GenericType(
                location=start.location,
                name=name.value,
                type_args=type_args
            )
        
        return SimpleType(location=start.location, name=name.value)
    
    def parse_function_type(self) -> FunctionType:
        """Parse a function type fn(A, B) -> C."""
        start = self.expect(TokenType.FN)
        self.expect(TokenType.LPAREN)
        
        param_types: list[TypeAnnotation] = []
        if not self.check(TokenType.RPAREN):
            param_types.append(self.parse_type())
            while self.match(TokenType.COMMA):
                param_types.append(self.parse_type())
        self.expect(TokenType.RPAREN)
        
        self.expect(TokenType.ARROW)
        return_type = self.parse_type()
        
        return FunctionType(
            location=start.location,
            param_types=param_types,
            return_type=return_type
        )
    
    # Expressions - Pratt parser for operators
    
    def parse_expression(self) -> Expression:
        """Parse an expression."""
        return self.parse_precedence(0)
    
    def parse_precedence(self, min_prec: int) -> Expression:
        """Parse expression with operator precedence (Pratt parsing)."""
        left = self.parse_unary()
        
        while True:
            op_token = self.current()
            
            if op_token.type not in PRECEDENCE:
                break
            
            prec = PRECEDENCE[op_token.type]
            if prec < min_prec:
                break
            
            self.advance()
            
            # Handle right associativity
            next_prec = prec + 1 if op_token.type not in RIGHT_ASSOCIATIVE else prec
            right = self.parse_precedence(next_prec)
            
            left = BinaryOp(
                location=left.location.extend_to(right.location),
                op=BINARY_OP_MAP[op_token.type],
                left=left,
                right=right
            )
        
        return left
    
    def parse_unary(self) -> Expression:
        """Parse unary expression."""
        if self.check(TokenType.MINUS, TokenType.NOT, TokenType.BANG, TokenType.TILDE):
            op_token = self.advance()
            operand = self.parse_unary()
            return UnaryOp(
                location=op_token.location.extend_to(operand.location),
                op=UNARY_OP_MAP[op_token.type],
                operand=operand
            )
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> Expression:
        """Parse postfix expressions (calls, indexing, field access)."""
        expr = self.parse_primary()
        
        while True:
            if self.match(TokenType.LPAREN):
                # Function call
                expr = self.parse_call_args(expr)
            elif self.match(TokenType.LBRACKET):
                # Indexing
                expr = self.parse_index(expr)
            elif self.match(TokenType.DOT):
                # Field access or method call
                field = self.expect(TokenType.IDENTIFIER, "Expected field name")
                if self.match(TokenType.LPAREN):
                    # Method call
                    args, kwargs = self.parse_args()
                    end = self.expect(TokenType.RPAREN, "Expected ')' after method arguments")
                    expr = MethodCall(
                        location=expr.location.extend_to(end.location),
                        receiver=expr,
                        method=field.value,
                        args=args,
                        kwargs=kwargs
                    )
                else:
                    expr = FieldAccess(
                        location=expr.location.extend_to(field.location),
                        base=expr,
                        field=field.value
                    )
            else:
                break
        
        return expr
    
    def parse_call_args(self, func: Expression) -> Call:
        """Parse function call arguments."""
        args, kwargs = self.parse_args()
        end = self.expect(TokenType.RPAREN, "Expected ')' after arguments")
        return Call(
            location=func.location.extend_to(end.location),
            func=func,
            args=args,
            kwargs=kwargs
        )
    
    def parse_args(self) -> tuple[list[Expression], list[tuple[str, Expression]]]:
        """Parse positional and keyword arguments."""
        args: list[Expression] = []
        kwargs: list[tuple[str, Expression]] = []
        
        while not self.check(TokenType.RPAREN, TokenType.EOF):
            # Check for keyword argument
            if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.ASSIGN:
                name = self.advance().value
                self.advance()  # =
                value = self.parse_expression()
                kwargs.append((name, value))
            else:
                if kwargs:
                    raise self.error(
                        "Positional argument after keyword argument",
                        ErrorCode.E106_INVALID_SYNTAX
                    )
                args.append(self.parse_expression())
            
            if not self.match(TokenType.COMMA):
                break
        
        return args, kwargs
    
    def parse_index(self, base: Expression) -> Index:
        """Parse indexing expression."""
        indices: list[Expression] = []
        
        if not self.check(TokenType.RBRACKET):
            indices.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                indices.append(self.parse_expression())
        
        end = self.expect(TokenType.RBRACKET, "Expected ']'")
        return Index(
            location=base.location.extend_to(end.location),
            base=base,
            indices=indices
        )
    
    def parse_primary(self) -> Expression:
        """Parse primary expression."""
        start = self.current()
        
        # Literals
        if self.check(TokenType.INT_LITERAL):
            token = self.advance()
            return Literal(location=token.location, value=token.value, kind='int')
        
        if self.check(TokenType.FLOAT_LITERAL):
            token = self.advance()
            return Literal(location=token.location, value=token.value, kind='float')
        
        if self.check(TokenType.STRING_LITERAL):
            token = self.advance()
            return Literal(location=token.location, value=token.value, kind='string')
        
        if self.check(TokenType.TRUE):
            token = self.advance()
            return Literal(location=token.location, value=True, kind='bool')
        
        if self.check(TokenType.FALSE):
            token = self.advance()
            return Literal(location=token.location, value=False, kind='bool')
        
        if self.check(TokenType.NONE):
            token = self.advance()
            return Literal(location=token.location, value=None, kind='none')
        
        # Identifiers
        if self.check(TokenType.IDENTIFIER):
            token = self.advance()
            return Identifier(location=token.location, name=token.value)
        
        # Grouped expression or tuple
        if self.match(TokenType.LPAREN):
            if self.check(TokenType.RPAREN):
                self.advance()
                return Literal(location=start.location, value=(), kind='unit')
            
            expr = self.parse_expression()
            
            if self.match(TokenType.COMMA):
                # Tuple
                elements = [expr]
                if not self.check(TokenType.RPAREN):
                    elements.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        if self.check(TokenType.RPAREN):
                            break
                        elements.append(self.parse_expression())
                self.expect(TokenType.RPAREN)
                return Tensor(location=start.location, elements=elements)
            
            self.expect(TokenType.RPAREN)
            return expr
        
        # Array/tensor literal
        if self.match(TokenType.LBRACKET):
            elements: list[Expression] = []
            if not self.check(TokenType.RBRACKET):
                elements.append(self.parse_expression())
                while self.match(TokenType.COMMA):
                    if self.check(TokenType.RBRACKET):
                        break
                    elements.append(self.parse_expression())
            end = self.expect(TokenType.RBRACKET)
            return Tensor(
                location=start.location.extend_to(end.location),
                elements=elements
            )
        
        # Block expression
        if self.check(TokenType.LBRACE):
            return self.parse_block()
        
        # If expression
        if self.check(TokenType.IF):
            return self.parse_if_expr()
        
        # Lambda
        if self.match(TokenType.PIPE):
            return self.parse_lambda(start)
        
        # param(expr) - inline parameter
        if self.match(TokenType.PARAM):
            if self.match(TokenType.LPAREN):
                init = self.parse_expression()
                end = self.expect(TokenType.RPAREN)
                return Param(
                    location=start.location.extend_to(end.location),
                    initializer=init
                )
            else:
                # Just the keyword, backtrack
                self.pos -= 1
        
        # obs(expr) - inline observation
        if self.match(TokenType.OBS):
            if self.match(TokenType.LPAREN):
                value = self.parse_expression()
                end = self.expect(TokenType.RPAREN)
                return Obs(
                    location=start.location.extend_to(end.location),
                    value=value
                )
            else:
                self.pos -= 1
        
        # rand distribution
        if self.match(TokenType.RAND):
            dist = self.parse_expression()
            return RandExpr(
                location=start.location.extend_to(dist.location),
                distribution=dist
            )
        
        # observe(value, distribution)
        if self.match(TokenType.OBSERVE):
            self.expect(TokenType.LPAREN)
            value = self.parse_expression()
            self.expect(TokenType.COMMA)
            dist = self.parse_expression()
            end = self.expect(TokenType.RPAREN)
            return ObserveExpr(
                location=start.location.extend_to(end.location),
                value=value,
                distribution=dist
            )
        
        # non_diff block
        if self.match(TokenType.NON_DIFF):
            body = self.parse_block()
            return NonDiffBlock(
                location=start.location.extend_to(body.location),
                body=body
            )
        
        raise self.error("Expected expression", ErrorCode.E102_EXPECTED_EXPRESSION)
    
    def parse_if_expr(self) -> IfExpr:
        """Parse an if expression with optional temperature."""
        start = self.expect(TokenType.IF)
        condition = self.parse_expression()
        
        # Optional temperature
        temperature: Optional[Expression] = None
        if self.match(TokenType.TEMPERATURE):
            temperature = self.parse_expression()
        
        self.skip_newlines()
        then_expr: Expression
        if self.check(TokenType.LBRACE):
            then_expr = self.parse_block()
        else:
            then_expr = self.parse_expression()
        
        self.skip_newlines()
        self.expect(TokenType.ELSE, "Expected 'else' in if expression")
        
        self.skip_newlines()
        else_expr: Expression
        if self.check(TokenType.LBRACE):
            else_expr = self.parse_block()
        elif self.check(TokenType.IF):
            else_expr = self.parse_if_expr()
        else:
            else_expr = self.parse_expression()
        
        return IfExpr(
            location=start.location.extend_to(else_expr.location),
            condition=condition,
            then_expr=then_expr,
            else_expr=else_expr,
            temperature=temperature
        )
    
    def parse_lambda(self, start: Token) -> Lambda:
        """Parse a lambda expression |params| expr."""
        params: list[Parameter] = []
        
        if not self.check(TokenType.PIPE):
            params.append(self.parse_parameter())
            while self.match(TokenType.COMMA):
                params.append(self.parse_parameter())
        
        self.expect(TokenType.PIPE)
        
        # Optional return type
        return_type: Optional[TypeAnnotation] = None
        if self.match(TokenType.ARROW):
            return_type = self.parse_type()
        
        body = self.parse_expression()
        
        return Lambda(
            location=start.location.extend_to(body.location),
            params=params,
            body=body,
            return_type=return_type
        )
