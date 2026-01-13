"""
SIR Builder - converts typed AST to Semantic IR.

This is the core lowering pass that transforms Delta AST into SIR,
handling:
- Desugaring if expressions to Gate/Mix
- Converting constraints to ConstraintOp
- Preserving role and effect information
- Setting up gradient requirements
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any

from delta.frontend.ast import (
    Module, Statement, Expression,
    FunctionDef, StructDef, LetStmt, AssignStmt, ExprStmt, ReturnStmt,
    IfStmt, WhileStmt, ForStmt, LearnBlock, ConstraintStmt as ASTConstraintStmt,
    ParamDecl, ObsDecl, ImportStmt,
    Block, IfExpr, BinaryOp, UnaryOp, Call, MethodCall,
    Index, FieldAccess, Tensor, Param, Obs, Identifier, Literal,
    Lambda, RandExpr, ObserveExpr, NonDiffBlock,
    BinaryOperator, UnaryOperator, ConstraintKind as ASTConstraintKind,
    Parameter, IdentifierPattern, TuplePattern, Pattern,
)
from delta.ir.sir import (
    SIRNode, SIRProperty, SIRModule, SIRFunction, SIRBlock,
    TensorOp, TensorOpKind, GateOp, MixOp, ConstraintOp, ConstraintKind,
    ParamRef, ObsRef, Const, StopGrad, Harden, GradBoundary,
    Layer, RandomVar, Observe, TupleIndex, ModeSwitch,
    BinaryTensorOp, UnaryTensorOp,
)
from delta.types.types import Type, FloatType, IntType, BoolType, TensorType, UnitType, AnyType
from delta.types.effects import EffectSet
from delta.types.roles import RoleInfo
from delta.types.inference import TypeInference, TypeEnvironment


# Map AST binary operators to SIR tensor ops
BINOP_MAP: dict[BinaryOperator, TensorOpKind] = {
    BinaryOperator.ADD: TensorOpKind.ADD,
    BinaryOperator.SUB: TensorOpKind.SUB,
    BinaryOperator.MUL: TensorOpKind.MUL,
    BinaryOperator.DIV: TensorOpKind.DIV,
    BinaryOperator.POW: TensorOpKind.POW,
    BinaryOperator.MATMUL: TensorOpKind.MATMUL,
    BinaryOperator.EQ: TensorOpKind.EQ,
    BinaryOperator.NE: TensorOpKind.NE,
    BinaryOperator.LT: TensorOpKind.LT,
    BinaryOperator.LE: TensorOpKind.LE,
    BinaryOperator.GT: TensorOpKind.GT,
    BinaryOperator.GE: TensorOpKind.GE,
}

# Map AST unary operators to SIR tensor ops
UNOP_MAP: dict[UnaryOperator, TensorOpKind] = {
    UnaryOperator.NEG: TensorOpKind.NEG,
}

# Map comparison operators for gates
COMPARISON_OPS: set[BinaryOperator] = {
    BinaryOperator.EQ, BinaryOperator.NE,
    BinaryOperator.LT, BinaryOperator.LE,
    BinaryOperator.GT, BinaryOperator.GE,
}


@dataclass
class BuilderContext:
    """Context for SIR building."""
    mode: str = "train"
    in_learn_block: bool = False
    in_non_diff: bool = False
    default_temperature: float = 1.0
    current_function: Optional[str] = None


class SIRBuilder:
    """
    Builds SIR from typed AST.
    
    Handles all the lowering transformations:
    - if with temperature -> Gate + Mix
    - Constraints -> ConstraintOp
    - param/obs -> ParamRef/ObsRef
    - Binary/unary ops -> TensorOp
    """
    
    def __init__(self, type_env: TypeEnvironment, inference: TypeInference) -> None:
        self.type_env = type_env
        self.inference = inference
        self.context = BuilderContext()
        self.bindings: dict[str, SIRNode] = {}
        self.constraints: list[ConstraintOp] = []
        self.function_defs: dict[str, FunctionDef] = {}  # Store function definitions for inlining
    
    def build_module(self, module: Module) -> SIRModule:
        """Build SIR module from AST module."""
        sir_module = SIRModule(name=module.name)
        
        for item in module.items:
            self._build_top_level(item, sir_module)
        
        # Add collected constraints
        for constraint in self.constraints:
            sir_module.add_constraint(constraint)
        
        return sir_module
    
    def _build_top_level(self, stmt: Statement, module: SIRModule) -> None:
        """Build a top-level item."""
        if isinstance(stmt, FunctionDef):
            # Store function definition for potential inlining
            self.function_defs[stmt.name] = stmt
            func = self._build_function(stmt)
            module.add_function(func)
        elif isinstance(stmt, ParamDecl):
            param = self._build_param_decl(stmt)
            module.add_param(stmt.name, param)
            self.bindings[stmt.name] = param
        elif isinstance(stmt, ObsDecl):
            obs = self._build_obs_decl(stmt)
            self.bindings[stmt.name] = obs
        elif isinstance(stmt, LearnBlock):
            self._build_learn_block(stmt, module)
        elif isinstance(stmt, ASTConstraintStmt):
            constraint = self._build_constraint_stmt(stmt)
            self.constraints.append(constraint)
        elif isinstance(stmt, LetStmt):
            node = self._build_expr(stmt.value)
            if isinstance(stmt.pattern, IdentifierPattern):
                self.bindings[stmt.pattern.name] = node
    
    def _build_function(self, func: FunctionDef) -> SIRFunction:
        """Build a SIR function."""
        old_context = self.context
        self.context = BuilderContext(current_function=func.name)
        
        # Build parameter info
        params: list[tuple[str, SIRProperty]] = []
        for param in func.params:
            typed = self.inference.infer_expression(
                Identifier(location=param.location, name=param.name),
                self.type_env
            )
            
            role = RoleInfo.const()
            if param.role == "param":
                role = RoleInfo.param()
            elif param.role == "obs":
                role = RoleInfo.obs()
            
            props = SIRProperty(
                dtype=typed.type,
                role=role,
                requires_grad=role.requires_grad,
                location=param.location
            )
            params.append((param.name, props))
            
            # Create binding for parameter
            # All function parameters are treated as inputs (like observations)
            if param.role == "param":
                self.bindings[param.name] = ParamRef(name=param.name, _props=props)
            else:
                # Regular function inputs and obs-annotated inputs use ObsRef
                self.bindings[param.name] = ObsRef(name=param.name, _props=props)
        
        # Build body
        body = self._build_block(func.body)
        
        # Get return type
        return_type = FloatType()
        if func.return_type:
            typed = self.inference._resolve_type_annotation(func.return_type, self.type_env)
            return_type = typed
        
        self.context = old_context
        
        return SIRFunction(
            name=func.name,
            params=params,
            body=body,
            return_type=return_type
        )
    
    def _build_learn_block(self, stmt: LearnBlock, module: SIRModule) -> None:
        """Build a learn block."""
        old_context = self.context
        self.context = BuilderContext(
            mode=stmt.mode,
            in_learn_block=True
        )
        
        try:
            # Build body statements
            for s in stmt.body.statements:
                self._build_statement(s)
            
            # Build result if present
            if stmt.body.result:
                self._build_expr(stmt.body.result)
                
            # Populate learn config in module
            from delta.ir.sir import SIRLearnConfig
            config = SIRLearnConfig(
                mode=stmt.mode,
                optimizer=stmt.optimizer,  # We'll share the AST expression for now
                epochs=self._eval_ast_int(stmt.epochs),
                batch_size=self._eval_ast_int(stmt.batch_size),
                location=stmt.location
            )
            module.add_learn_config(config)
        finally:
            self.context = old_context
            
    def _eval_ast_int(self, expr: Optional[Expression]) -> Optional[int]:
        """Try to evaluate an AST expression as an integer literal."""
        from delta.frontend.ast import Literal
        if isinstance(expr, Literal) and expr.kind == 'int':
            return int(expr.value)
        return None
    
    def _build_param_decl(self, stmt: ParamDecl) -> ParamRef:
        """Build a param declaration."""
        typed = self.inference.infer_expression(
            Identifier(location=stmt.location, name=stmt.name),
            self.type_env
        ) if self.type_env.lookup(stmt.name) else None
        
        props = SIRProperty(
            dtype=typed.type if typed else TensorType(FloatType()),
            role=RoleInfo.param(),
            requires_grad=True,
            location=stmt.location
        )
        
        return ParamRef(name=stmt.name, _props=props)
    
    def _build_obs_decl(self, stmt: ObsDecl) -> ObsRef:
        """Build an obs declaration."""
        typed = self.inference.infer_expression(
            Identifier(location=stmt.location, name=stmt.name),
            self.type_env
        ) if self.type_env.lookup(stmt.name) else None
        
        props = SIRProperty(
            dtype=typed.type if typed else TensorType(FloatType()),
            role=RoleInfo.obs(),
            requires_grad=False,
            location=stmt.location
        )
        
        return ObsRef(name=stmt.name, _props=props)
    
    def _build_constraint_stmt(self, stmt: ASTConstraintStmt) -> ConstraintOp:
        """Build a constraint statement."""
        expr_node = self._build_expr(stmt.expr)
        
        # Determine constraint kind
        if stmt.kind == ASTConstraintKind.REQUIRE:
            kind = ConstraintKind.EQUALITY
        elif stmt.kind == ASTConstraintKind.PREFER:
            kind = ConstraintKind.INEQUALITY
        else:
            kind = ConstraintKind.LIKELIHOOD
        
        # Build weight (default 1.0)
        if stmt.weight:
            weight = self._build_expr(stmt.weight)
        else:
            weight = Const(value=1.0)
        
        # Build slack if present
        slack = None
        if stmt.slack:
            slack = self._build_expr(stmt.slack)
        
        props = SIRProperty(
            dtype=ConstraintType(),
            role=RoleInfo.constraint(),
            location=stmt.location
        )
        
        return ConstraintOp(
            kind=kind,
            lhs=expr_node,
            rhs=None,  # All constraints normalized to expr checks
            weight=weight,
            slack=slack,
            _props=props
        )
    
    def _build_statement(self, stmt: Statement) -> Optional[SIRNode]:
        """Build a statement."""
        if isinstance(stmt, LetStmt):
            node = self._build_expr(stmt.value)
            self._bind_pattern(stmt.pattern, node)
            return node
        elif isinstance(stmt, ExprStmt):
            return self._build_expr(stmt.expr)
        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                return self._build_expr(stmt.value)
            return None
        elif isinstance(stmt, ASTConstraintStmt):
            constraint = self._build_constraint_stmt(stmt)
            self.constraints.append(constraint)
            return constraint
        elif isinstance(stmt, ParamDecl):
            param = self._build_param_decl(stmt)
            self.bindings[stmt.name] = param
            return param
        elif isinstance(stmt, ObsDecl):
            obs = self._build_obs_decl(stmt)
            self.bindings[stmt.name] = obs
            return obs
        elif isinstance(stmt, IfStmt):
            return self._build_if_stmt(stmt)
        elif isinstance(stmt, AssignStmt):
            node = self._build_expr(stmt.value)
            self._bind_pattern(stmt.target, node)
            return node
        elif isinstance(stmt, ForStmt):
            return self._build_for_stmt(stmt)
        else:
            return None
    
    def _build_if_stmt(self, stmt: IfStmt) -> Optional[SIRNode]:
        """Build an if statement."""
        # Support optional temperature (soft-conditional)
        temperature = None
        if stmt.temperature:
            temperature = self._build_expr(stmt.temperature)
        elif self.context.mode == "train" and self.context.in_learn_block:
            temperature = Const(value=1.0) # Default temperature for learn blocks
            
        condition = self._build_expr(stmt.condition)
        then_block = self._build_block(stmt.then_block)
        
        else_block = None
        if stmt.else_block:
            if isinstance(stmt.else_block, Block):
                else_block = self._build_block(stmt.else_block)
            else:
                # Elif chain
                else_sir = self._build_if_stmt(stmt.else_block)
                if else_sir:
                    else_block = SIRBlock(nodes=[else_sir], result=else_sir)
        
        # If it's a soft conditional, return a MixOp to be part of the block's results
        if temperature:
            gate = GateOp(compare=TensorOpKind.GT, lhs=condition, rhs=Const(value=0.0), temperature=temperature)
            then_val = then_block.result or Const(value=0.0)
            else_val = (else_block.result if else_block else None) or Const(value=0.0)
            return MixOp(gate, then_val, else_val)
        
        return None

    def _bind_pattern(self, pattern: Pattern | Expression, node: SIRNode) -> None:
        """Bind names from a pattern to a SIR node (with destructuring)."""
        if isinstance(pattern, IdentifierPattern):
            self.bindings[pattern.name] = node
        elif isinstance(pattern, Identifier):
             self.bindings[pattern.name] = node
        elif isinstance(pattern, TuplePattern):
            for i, p in enumerate(pattern.elements):
                # Extract i-th element from node
                extracted = TupleIndex(operand=node, index=i)
                self._bind_pattern(p, extracted)
        elif isinstance(pattern, list):  # For potential literal lists used as patterns
            for i, p in enumerate(pattern):
                extracted = TupleIndex(operand=node, index=i)
                self._bind_pattern(p, extracted)

    def _build_for_stmt(self, stmt: ForStmt) -> list[SIRNode]:
        """Build a for statement (unrolled)."""
        # We only support unrolling for certain iterables: list literals, range()
        iterable = stmt.iterable
        elements = []
        
        if isinstance(iterable, Tensor):
            # List literal [1, 2, 3]
            elements = iterable.elements
        elif isinstance(iterable, Call) and isinstance(iterable.func, Identifier) and iterable.func.name == "range":
            # range(N) or range(start, stop)
            args = []
            for arg in iterable.args:
                val = self._build_expr(arg)
                if isinstance(val, Const):
                    args.append(val.value)
                else:
                    return []
            
            if len(args) == 1:
                elements = [Literal(location=iterable.location, value=i, kind='int') for i in range(args[0])]
            elif len(args) == 2:
                elements = [Literal(location=iterable.location, value=i, kind='int') for i in range(args[0], args[1])]
            elif len(args) == 3:
                elements = [Literal(location=iterable.location, value=i, kind='int') for i in range(args[0], args[1], args[2])]
        
        if not elements:
            return []
            
        unrolled_nodes = []
        # Unroll the loop
        for elem in elements:
            # Bind the loop variable
            if isinstance(stmt.pattern, IdentifierPattern):
                # Build the element expression
                elem_node = self._build_expr(elem)
                old_binding = self.bindings.get(stmt.pattern.name)
                self.bindings[stmt.pattern.name] = elem_node
                
                # Build the body
                body_block = self._build_block(stmt.body)
                unrolled_nodes.extend(body_block.nodes)
                if body_block.result:
                    unrolled_nodes.append(body_block.result)
                
                # Restore binding
                if old_binding:
                    self.bindings[stmt.pattern.name] = old_binding
                else:
                    del self.bindings[stmt.pattern.name]
        
        return unrolled_nodes
    
    def _build_block(self, block: Block) -> SIRBlock:
        """Build a block."""
        nodes: list[SIRNode] = []
        
        for stmt in block.statements:
            node = self._build_statement(stmt)
            if isinstance(node, list):
                nodes.extend(node)
            elif node:
                nodes.append(node)
        
        result = None
        if block.result:
            result = self._build_expr(block.result)
            nodes.append(result)
        elif nodes and not isinstance(block.statements[-1], (ParamDecl, ObsDecl)):
            # Fallback: if no explicit result, use the last node as result (like Rust/Ruby)
            # Declared params/obs don't produce values
            result = nodes[-1]
        
        return SIRBlock(nodes=nodes, result=result)
    
    def _build_expr(self, expr: Expression) -> SIRNode:
        """Build an expression."""
        if isinstance(expr, Literal):
            return self._build_literal(expr)
        elif isinstance(expr, Identifier):
            return self._build_identifier(expr)
        elif isinstance(expr, BinaryOp):
            return self._build_binary_op(expr)
        elif isinstance(expr, UnaryOp):
            return self._build_unary_op(expr)
        elif isinstance(expr, Call):
            return self._build_call(expr)
        elif isinstance(expr, MethodCall):
            return self._build_method_call(expr)
        elif isinstance(expr, Index):
            return self._build_index(expr)
        elif isinstance(expr, FieldAccess):
            return self._build_field_access(expr)
        elif isinstance(expr, IfExpr):
            return self._build_if_expr(expr)
        elif isinstance(expr, Block):
            block = self._build_block(expr)
            return block.result or Const(value=None)
        elif isinstance(expr, Lambda):
            return self._build_lambda(expr)
        elif isinstance(expr, Tensor):
            return self._build_tensor_literal(expr)
        elif isinstance(expr, Param):
            return self._build_param_expr(expr)
        elif isinstance(expr, Obs):
            return self._build_obs_expr(expr)
        elif isinstance(expr, RandExpr):
            return self._build_rand_expr(expr)
        elif isinstance(expr, ObserveExpr):
            return self._build_observe_expr(expr)
        elif isinstance(expr, NonDiffBlock):
            return self._build_non_diff_block(expr)
        else:
            return Const(value=None)
    
    def _build_literal(self, lit: Literal) -> Const:
        """Build a literal."""
        if lit.kind == 'int':
            dtype = IntType()
        elif lit.kind == 'float':
            dtype = FloatType()
        elif lit.kind == 'bool':
            dtype = BoolType()
        else:
            dtype = UnitType()
        
        props = SIRProperty(dtype=dtype, requires_grad=False)
        return Const(value=lit.value, _props=props)
    
    def _build_identifier(self, ident: Identifier) -> SIRNode:
        """Build an identifier reference."""
        if ident.name in self.bindings:
            return self.bindings[ident.name]
        
        # Check type environment
        binding = self.type_env.lookup(ident.name)
        if binding:
            props = SIRProperty(
                dtype=binding.type,
                role=binding.role,
                requires_grad=binding.role.requires_grad,
                location=ident.location
            )
            
            if binding.role.primary_role.name == "PARAM":
                node = ParamRef(name=ident.name, _props=props)
            else:
                # All other references (including CONST for function params) are ObsRef
                node = ObsRef(name=ident.name, _props=props)
            
            self.bindings[ident.name] = node
            return node
        
        # Unknown - create const placeholder
        return Const(value=None, _props=SIRProperty(location=ident.location))
    
    def _build_binary_op(self, op: BinaryOp) -> SIRNode:
        """Build a binary operation."""
        left = self._build_expr(op.left)
        right = self._build_expr(op.right)
        
        # Logical operators - convert to soft logic in train mode
        if op.op == BinaryOperator.AND:
            # Soft AND: a * b
            return BinaryTensorOp(TensorOpKind.MUL, left, right)
        elif op.op == BinaryOperator.OR:
            # Soft OR: 1 - (1-a) * (1-b)
            one = Const(value=1.0)
            one_minus_left = BinaryTensorOp(TensorOpKind.SUB, one, left)
            one_minus_right = BinaryTensorOp(TensorOpKind.SUB, one, right)
            product = BinaryTensorOp(TensorOpKind.MUL, one_minus_left, one_minus_right)
            return BinaryTensorOp(TensorOpKind.SUB, one, product)
        elif op.op == BinaryOperator.POW:
            return BinaryTensorOp(TensorOpKind.POW, left, right)
        elif op.op == BinaryOperator.OBSERVE:
            # x ~ dist -> Observe(x, dist)
            # Find the distribution name and params from the right side
            if isinstance(right, RandomVar):
                return Observe(value=left, distribution=right.distribution, params=right.params)
            elif isinstance(right, Const) and isinstance(right.value, tuple):
                # Unpack (func_name, args, kwargs) from distribution Const
                if len(right.value) == 3:
                    dist_name, dist_args, dist_kwargs = right.value
                else:
                    dist_name, dist_args = right.value[:2]
                    dist_kwargs = {}
                return Observe(value=left, distribution=dist_name, params=dist_args, kwargs=dist_kwargs)
            elif isinstance(right, Call) and isinstance(right.func, Identifier):
                # For case where Normal(0, 1) hasn't been lowered to RandomVar yet
                return Observe(value=left, distribution=right.func.name, params=[self._build_expr(a) for a in right.args])
            else:
                # print(f"DEBUG: OBSERVE unknown right={type(right)}")
                return Observe(value=left, distribution="Unknown", params=[right])
        
        # print(f"DEBUG: BinaryOp op={op.op}, left={type(left)}, right={type(right)}")
        # Comparison operators
        compare_map = {
            BinaryOperator.EQ: TensorOpKind.EQ,
            BinaryOperator.NE: TensorOpKind.NE,
            BinaryOperator.LT: TensorOpKind.LT,
            BinaryOperator.LE: TensorOpKind.LE,
            BinaryOperator.GT: TensorOpKind.GT,
            BinaryOperator.GE: TensorOpKind.GE,
        }
        if op.op in compare_map:
            return BinaryTensorOp(compare_map[op.op], left, right)
        
        # Map to tensor op
        if op.op in BINOP_MAP:
            tensor_op = BINOP_MAP[op.op]
            return BinaryTensorOp(tensor_op, left, right)
        
        # Fallback
        return BinaryTensorOp(TensorOpKind.ADD, left, right)
    
    def _build_unary_op(self, op: UnaryOp) -> SIRNode:
        """Build a unary operation."""
        operand = self._build_expr(op.operand)
        
        if op.op == UnaryOperator.NOT:
            # Soft NOT: 1 - x
            one = Const(value=1.0)
            return BinaryTensorOp(TensorOpKind.SUB, one, operand)
        elif op.op in UNOP_MAP:
            return UnaryTensorOp(UNOP_MAP[op.op], operand)
        
        return operand
    
    def _build_call(self, call: Call) -> SIRNode:
        """Build a function call."""
        # Build arguments
        args = [self._build_expr(arg) for arg in call.args]
        kwargs = {k: self._build_expr(v) for k, v in getattr(call, 'kwargs', [])}
        
        # Get function name
        if isinstance(call.func, Identifier):
            func_name = call.func.name
            
            # ─────────────────────────────────────────────────────────────────
            # Unary math functions
            # ─────────────────────────────────────────────────────────────────
            unary_ops = {
                'exp': TensorOpKind.EXP,
                'log': TensorOpKind.LOG,
                'sin': TensorOpKind.SIN,
                'cos': TensorOpKind.COS,
                'tanh': TensorOpKind.TANH,
                'sigmoid': TensorOpKind.SIGMOID,
                'relu': TensorOpKind.RELU,
                'softmax': TensorOpKind.SOFTMAX,
                'sqrt': TensorOpKind.SQRT,
                'abs': TensorOpKind.ABS,
                'neg': TensorOpKind.NEG,
                'gelu': TensorOpKind.GELU,
            }
            if func_name in unary_ops:
                return UnaryTensorOp(unary_ops[func_name], args[0])
            
            # ─────────────────────────────────────────────────────────────────
            # Gradient control functions
            # ─────────────────────────────────────────────────────────────────
            if func_name == 'StopGrad':
                if not args:
                    return Const(value=None)
                props = SIRProperty(
                    dtype=args[0].dtype,
                    shape=args[0].shape,
                    requires_grad=False,
                    location=call.location if hasattr(call, 'location') else None
                )
                return StopGrad(operand=args[0], _props=props)
            
            if func_name == 'Harden':
                if not args:
                    return Const(value=None)
                threshold = 0.5
                if len(args) > 1 and isinstance(args[1], Const):
                    threshold = args[1].value
                props = SIRProperty(
                    dtype=BoolType(),
                    requires_grad=False,
                    location=call.location if hasattr(call, 'location') else None
                )
                return Harden(operand=args[0], threshold=threshold, _props=props)
            
            # ─────────────────────────────────────────────────────────────────
            # Reduction functions (variadic - tensor + optional kwargs)
            # ─────────────────────────────────────────────────────────────────
            reductions = {
                'sum': TensorOpKind.SUM,
                'mean': TensorOpKind.MEAN,
                'max': TensorOpKind.MAX,
                'min': TensorOpKind.MIN,
                'prod': TensorOpKind.PROD,
            }
            if func_name in reductions:
                return TensorOp(
                    op=reductions[func_name],
                    operands=args,
                    _props=SIRProperty(dtype=FloatType())
                )
            
            # ─────────────────────────────────────────────────────────────────
            # Initializers
            # ─────────────────────────────────────────────────────────────────
            initializers = {
                'zeros': TensorOpKind.ZEROS,
                'ones': TensorOpKind.ONES,
                'rand': TensorOpKind.RAND,
                'randn': TensorOpKind.RANDN,
                'full': TensorOpKind.FULL,
                'eye': TensorOpKind.EYE,
            }
            if func_name in initializers:
                # Special handling: if first arg is a list literal (STACK) or a variable
                # bound to one, unpack it as dimensions for the initializer.
                new_args = []
                for i, arg in enumerate(args):
                    if i == 0:
                        if isinstance(arg, TensorOp) and arg.op == TensorOpKind.STACK:
                            # Unpack dynamic shape list
                            new_args.extend(arg.operands)
                        elif isinstance(arg, Const) and isinstance(arg.value, (list, tuple)):
                            # Unpack constant list/tuple shape
                            new_args.extend([Const(value=v) for v in arg.value])
                        elif isinstance(arg, Const) and hasattr(arg.value, 'tolist'):
                            # Unpack constant tensor shape
                            import torch
                            if isinstance(arg.value, torch.Tensor):
                                values = arg.value.tolist()
                                if isinstance(values, list):
                                    new_args.extend([Const(value=v) for v in values])
                                else:
                                    new_args.append(arg)
                            else:
                                new_args.append(arg)
                        else:
                            new_args.append(arg)
                    else:
                        new_args.append(arg)
                
                return TensorOp(
                    op=initializers[func_name],
                    operands=new_args,
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            # ─────────────────────────────────────────────────────────────────
            # Binary operations
            # ─────────────────────────────────────────────────────────────────
            if func_name == 'matmul':
                return BinaryTensorOp(TensorOpKind.MATMUL, args[0], args[1])
            
            if func_name == 'pow':
                return BinaryTensorOp(TensorOpKind.POW, args[0], args[1] if len(args) > 1 else Const(value=2.0))
            
            # ─────────────────────────────────────────────────────────────────
            # Shape operations
            # ─────────────────────────────────────────────────────────────────
            if func_name == 'reshape':
                # Special handling for reshape(tensor, [dims...])
                new_args = [args[0]]
                if len(args) > 1:
                    shape_arg = args[1]
                    if isinstance(shape_arg, TensorOp) and shape_arg.op == TensorOpKind.STACK:
                        new_args.extend(shape_arg.operands)
                    elif isinstance(shape_arg, Const) and isinstance(shape_arg.value, (list, tuple)):
                        new_args.extend([Const(value=v) for v in shape_arg.value])
                    else:
                        new_args.append(shape_arg)
                    
                    # Append remaining args if any (e.g. reshape(x, shape, extra??))
                    new_args.extend(args[2:])
                
                return TensorOp(
                    op=TensorOpKind.RESHAPE,
                    operands=new_args,
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            if func_name == 'transpose':
                return TensorOp(
                    op=TensorOpKind.TRANSPOSE,
                    operands=args,
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            if func_name == 'squeeze':
                return TensorOp(
                    op=TensorOpKind.SQUEEZE,
                    operands=args,
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            if func_name == 'unsqueeze':
                return TensorOp(
                    op=TensorOpKind.UNSQUEEZE,
                    operands=args,
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            if func_name == 'flatten':
                return TensorOp(
                    op=TensorOpKind.FLATTEN,
                    operands=args,
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            if func_name in ('cat', 'concat'):
                dim = 0
                if 'dim' in kwargs:
                    dim_node = kwargs['dim']
                    if isinstance(dim_node, Const):
                        dim = dim_node.value
                return TensorOp(
                    op=TensorOpKind.CONCAT,
                    operands=args,
                    attrs={'dim': dim},
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            if func_name == 'stack':
                dim = 0
                if 'dim' in kwargs:
                    dim_node = kwargs['dim']
                    if isinstance(dim_node, Const):
                        dim = dim_node.value
                return TensorOp(
                    op=TensorOpKind.STACK,
                    operands=args,
                    attrs={'dim': dim},
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            # ─────────────────────────────────────────────────────────────────
            # Neural network layers (Constructors)
            # ─────────────────────────────────────────────────────────────────
            layers = {
                'Linear', 'Conv1d', 'Conv2d', 'BatchNorm1d', 'BatchNorm2d',
                'LayerNorm', 'Dropout', 'Embedding', 'LSTM', 'GRU', 'RNN',
                'MultiheadAttention'
            }
            if func_name in layers:
                return Layer(
                    kind=func_name,
                    args=[getattr(a, 'value', a) for a in args], # Extract scalar values if possible
                    _props=SIRProperty(dtype=AnyType()) # Layers return a callable object
                )

            if func_name == 'embedding':
                return TensorOp(
                    op=TensorOpKind.EMBEDDING,
                    operands=args,
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            if func_name == 'cross_entropy':
                return TensorOp(
                    op=TensorOpKind.CROSS_ENTROPY,
                    operands=args,
                    _props=SIRProperty(dtype=FloatType())
                )
            
            if func_name == 'mse_loss':
                return TensorOp(
                    op=TensorOpKind.MSE_LOSS,
                    operands=args,
                    _props=SIRProperty(dtype=FloatType())
                )
            
            if func_name == 'causal_mask':
                return TensorOp(
                    op=TensorOpKind.CAUSAL_MASK,
                    operands=args,
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            # ─────────────────────────────────────────────────────────────────
            # Shape queries
            # ─────────────────────────────────────────────────────────────────
            if func_name == 'shape':
                return TensorOp(
                    op=TensorOpKind.SHAPE,
                    operands=args,
                    _props=SIRProperty(dtype=IntType())
                )
            
            if func_name == 'slice':
                return TensorOp(
                    op=TensorOpKind.SLICE,
                    operands=args,
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            # ─────────────────────────────────────────────────────────────────
            # Comparison
            # ─────────────────────────────────────────────────────────────────
            if func_name == 'where':
                return TensorOp(
                    op=TensorOpKind.WHERE,
                    operands=args,
                    _props=SIRProperty(dtype=TensorType(FloatType()))
                )
            
            if func_name == 'argmax':
                return TensorOp(
                    op=TensorOpKind.ARGMAX,
                    operands=args,
                    _props=SIRProperty(dtype=TensorType(IntType()))
                )
            
            # ─────────────────────────────────────────────────────────────────
            # Distribution constructors
            # ─────────────────────────────────────────────────────────────────
            if func_name in ('Normal', 'Bernoulli', 'Categorical', 'Uniform', 'Beta', 'Gamma', 'Exponential', 'Poisson', 'Dirichlet', 'MultivariateNormal'):
                return Const(value=(func_name, args, kwargs))
            
            # ─────────────────────────────────────────────────────────────────
            # User-defined function call - try to inline
            # ─────────────────────────────────────────────────────────────────
            if func_name in self.function_defs:
                return self._inline_function_call(func_name, args, call)
            
            # ─────────────────────────────────────────────────────────────────
            # Variable call (e.g., fc(input))
            # ─────────────────────────────────────────────────────────────────
            if func_name in self.bindings:
                bound_node = self.bindings[func_name]
                if isinstance(bound_node, Layer):
                    return TensorOp(
                        op=TensorOpKind.CALL_LAYER,
                        operands=[bound_node] + args,
                        _props=SIRProperty(dtype=TensorType(FloatType()))
                    )
        
        # Generic call - represented as const for now
        return Const(value=None)
    
    def _inline_function_call(self, func_name: str, args: list[SIRNode], call: Call) -> SIRNode:
        """Inline a user-defined function call."""
        func_def = self.function_defs[func_name]
        
        # Create new bindings for function parameters
        old_bindings = self.bindings.copy()
        
        for param, arg in zip(func_def.params, args):
            self.bindings[param.name] = arg
        
        # Build function body
        result = self._build_block(func_def.body)
        
        # Restore old bindings
        self.bindings = old_bindings
        
        # Return the block result
        return result.result if result.result else Const(value=None)
    
    def _build_method_call(self, call: MethodCall) -> SIRNode:
        """Build a method call."""
        receiver = self._build_expr(call.receiver)
        args = [self._build_expr(arg) for arg in call.args]
        
        # Handle common tensor methods
        method = call.method
        if method in ('sum', 'mean', 'max', 'min'):
            op_map = {
                'sum': TensorOpKind.SUM,
                'mean': TensorOpKind.MEAN,
                'max': TensorOpKind.MAX,
                'min': TensorOpKind.MIN,
            }
            return TensorOp(op=op_map[method], operands=[receiver] + args)
        elif method in ('reshape', 'view'):
            return TensorOp(op=TensorOpKind.RESHAPE, operands=[receiver] + args)
        elif method in ('transpose', 'T'):
            return TensorOp(op=TensorOpKind.TRANSPOSE, operands=[receiver])
        
        return receiver
    
    def _build_index(self, index: Index) -> SIRNode:
        """Build an indexing operation."""
        base = self._build_expr(index.base)
        indices = [self._build_expr(i) for i in index.indices]
        
        return TensorOp(
            op=TensorOpKind.SLICE,
            operands=[base] + indices,
            _props=SIRProperty(dtype=base.dtype)
        )
    
    def _build_field_access(self, access: FieldAccess) -> SIRNode:
        """Build a field access."""
        base = self._build_expr(access.base)
        # For now, just return base - proper struct handling would extract field
        return base
    
    def _build_if_expr(self, expr: IfExpr) -> SIRNode:
        """
        Build an if expression.
        
        If temperature is specified: Gate + Mix (differentiable)
        Otherwise: Hard conditional (in non_diff mode only)
        """
        cond = self._build_expr(expr.condition)
        then_val = self._build_expr(expr.then_expr)
        else_val = self._build_expr(expr.else_expr)
        
        if expr.temperature or (self.context.mode == "train" and self.context.in_learn_block):
            # Differentiable conditional using Gate/Mix
            if expr.temperature:
                temp = self._build_expr(expr.temperature)
            else:
                temp = Const(value=self.context.default_temperature)
            
            # If condition is already a comparison, create gate from it
            # Otherwise, gate against 0.5
            if isinstance(cond, TensorOp) and cond.op in (
                TensorOpKind.LT, TensorOpKind.LE, TensorOpKind.GT, 
                TensorOpKind.GE, TensorOpKind.EQ, TensorOpKind.NE
            ):
                gate = GateOp(
                    compare=cond.op,
                    lhs=cond.operands[0],
                    rhs=cond.operands[1],
                    temperature=temp,
                    _props=SIRProperty(dtype=FloatType(), requires_grad=True)
                )
            else:
                # Treat condition as probability
                threshold = Const(value=0.5)
                gate = GateOp(
                    compare=TensorOpKind.GT,
                    lhs=cond,
                    rhs=threshold,
                    temperature=temp,
                    _props=SIRProperty(dtype=FloatType(), requires_grad=True)
                )
            
            # Mix the branches
            props = SIRProperty(
                dtype=then_val.dtype,
                requires_grad=then_val.requires_grad or else_val.requires_grad
            )
            return MixOp(gate=gate, then_value=then_val, else_value=else_val, _props=props)
        else:
            # Hard conditional - for infer mode or non_diff
            # Use ModeSwitch to select behavior
            if self.context.mode == "infer":
                # In infer mode, use hard selection
                threshold = Const(value=0.5)
                gate = GateOp(
                    compare=TensorOpKind.GT,
                    lhs=cond,
                    rhs=threshold,
                    temperature=Const(value=0.01),  # Very low temp = hard
                    _props=SIRProperty(dtype=FloatType())
                )
                hardened = Harden(operand=gate)
                props = SIRProperty(dtype=then_val.dtype, requires_grad=False)
                return MixOp(gate=hardened, then_value=then_val, else_value=else_val, _props=props)
            else:
                # Default to soft
                temp = Const(value=1.0)
                threshold = Const(value=0.5)
                gate = GateOp(
                    compare=TensorOpKind.GT,
                    lhs=cond,
                    rhs=threshold,
                    temperature=temp,
                    _props=SIRProperty(dtype=FloatType(), requires_grad=True)
                )
                props = SIRProperty(
                    dtype=then_val.dtype,
                    requires_grad=then_val.requires_grad or else_val.requires_grad
                )
                return MixOp(gate=gate, then_value=then_val, else_value=else_val, _props=props)
    
    def _build_lambda(self, lam: Lambda) -> SIRNode:
        """Build a lambda expression."""
        # Lambdas are represented as closures
        # For now, just build the body
        return self._build_expr(lam.body)
    
    def _build_tensor_literal(self, tensor: Tensor) -> SIRNode:
        """Build a tensor literal."""
        elements = [self._build_expr(e) for e in tensor.elements]
        
        if not elements:
            return Const(value=[])
        
        # If all elements are constants, keep it as Const for backward compatibility/optimization
        if all(isinstance(e, Const) for e in elements):
            values = [e.value for e in elements]
            return Const(value=values)
            
        # Otherwise, stack them
        # We assume the first element defines the dtype and shape (of each element)
        return TensorOp(
            op=TensorOpKind.STACK,
            operands=elements,
            attrs={'dim': 0},
            _props=SIRProperty(
                dtype=TensorType(elements[0].dtype),
                requires_grad=any(e.requires_grad for e in elements)
            )
        )
    
    def _build_param_expr(self, param: Param) -> ParamRef:
        """Build a param expression."""
        init = self._build_expr(param.initializer)
        
        # Create anonymous param
        name = f"_param_{param.initializer.location.line}_{param.initializer.location.column}"
        
        props = SIRProperty(
            dtype=init.dtype,
            role=RoleInfo.param(),
            requires_grad=True,
            location=param.location
        )
        
        return ParamRef(name=name, _props=props)
    
    def _build_obs_expr(self, obs: Obs) -> ObsRef:
        """Build an obs expression."""
        value = self._build_expr(obs.value)
        
        # Create anonymous obs
        name = f"_obs_{obs.value.location.line}_{obs.value.location.column}"
        
        props = SIRProperty(
            dtype=value.dtype,
            role=RoleInfo.obs(),
            requires_grad=False,
            location=obs.location
        )
        
        return ObsRef(name=name, _props=props)
    
    def _build_rand_expr(self, rand: RandExpr) -> RandomVar:
        """Build a rand expression."""
        dist = self._build_expr(rand.distribution)
        
        # Extract distribution info
        if isinstance(dist, Const) and isinstance(dist.value, tuple):
            if len(dist.value) == 3:
                dist_name, params, kwargs = dist.value
            else:
                dist_name, params = dist.value[:2]
                kwargs = {}
        else:
            dist_name = "Normal"
            params = [Const(value=0.0), Const(value=1.0)]
            kwargs = {}
        
        props = SIRProperty(
            dtype=FloatType(),
            effects=EffectSet.stochastic(),
            requires_grad=True,  # Reparameterized
            location=rand.location
        )
        
        return RandomVar(distribution=dist_name, params=params, kwargs=kwargs, _props=props)
    
    def _build_observe_expr(self, obs: ObserveExpr) -> Observe:
        """Build an observe expression."""
        value = self._build_expr(obs.value)
        dist = self._build_expr(obs.distribution)
        
        if isinstance(dist, Const) and isinstance(dist.value, tuple):
            if len(dist.value) == 3:
                dist_name, params, kwargs = dist.value
            else:
                dist_name, params = dist.value[:2]
                kwargs = {}
        else:
            dist_name = "Normal"
            params = [Const(value=0.0), Const(value=1.0)]
            kwargs = {}
        
        props = SIRProperty(
            dtype=FloatType(),
            effects=EffectSet.stochastic(),
            location=obs.location
        )
        
        return Observe(value=value, distribution=dist_name, params=params, kwargs=kwargs, _props=props)
    
    def _build_non_diff_block(self, block: NonDiffBlock) -> SIRNode:
        """Build a non_diff block."""
        old_in_non_diff = self.context.in_non_diff
        self.context.in_non_diff = True
        
        try:
            sir_block = self._build_block(block.body)
            result = sir_block.result or Const(value=None)
            
            # Wrap in StopGrad
            return StopGrad(operand=result)
        finally:
            self.context.in_non_diff = old_in_non_diff


def build_sir(module: Module, type_env: TypeEnvironment, inference: TypeInference) -> SIRModule:
    """Convenience function to build SIR from AST."""
    builder = SIRBuilder(type_env, inference)
    return builder.build_module(module)


class SIRGraphBuilder:
    """
    Simple graph builder for creating SIR graphs in tests.
    
    Provides a convenient API for building SIR graphs without 
    going through the full AST/type inference pipeline.
    """
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.inputs: list[tuple[str, SIRProperty]] = []
        self.params: dict[str, ParamRef] = {}
        self.ops: list[TensorOp] = []
        self.outputs: list[TensorOp] = []
        self._node_counter = 0
    
    def _make_props(self, dtype: str = "float32", shape: tuple = ()) -> SIRProperty:
        return SIRProperty(
            dtype=TensorType(FloatType()) if dtype == "float32" else FloatType(),
            role=RoleInfo.const(),
            requires_grad=False
        )
    
    def add_input(self, name: str, dtype: str = "float32", shape: tuple = ()) -> TensorOp:
        """Add an input to the graph."""
        props = self._make_props(dtype, shape)
        props.role = RoleInfo.obs()
        op = TensorOp(op=TensorOpKind.IDENTITY, operands=[], _props=props)
        op._name = name
        self.inputs.append((name, props))
        return op
    
    def add_param(self, name: str, dtype: str = "float32", shape: tuple = ()) -> ParamRef:
        """Add a parameter to the graph."""
        props = self._make_props(dtype, shape)
        props.role = RoleInfo.param()
        props.requires_grad = True
        param = ParamRef(name=name, _props=props)
        self.params[name] = param
        return param
    
    def add_tensor_op(self, op_name: str, inputs: list, dtype: str = "float32", shape: tuple = ()) -> TensorOp:
        """Add a tensor operation to the graph."""
        op_map = {
            "relu": TensorOpKind.RELU,
            "sigmoid": TensorOpKind.SIGMOID,
            "tanh": TensorOpKind.TANH,
            "exp": TensorOpKind.EXP,
            "log": TensorOpKind.LOG,
            "matmul": TensorOpKind.MATMUL,
            "add": TensorOpKind.ADD,
            "sub": TensorOpKind.SUB,
            "mul": TensorOpKind.MUL,
            "div": TensorOpKind.DIV,
            "sum": TensorOpKind.SUM,
            "mean": TensorOpKind.MEAN,
            "square": TensorOpKind.POW,  # Will use x**2
        }
        kind = op_map.get(op_name, TensorOpKind.IDENTITY)
        props = self._make_props(dtype, shape)
        op = TensorOp(op=kind, operands=inputs, _props=props)
        self.ops.append(op)
        return op
    
    def add_output(self, node) -> None:
        """Mark a node as an output."""
        self.outputs.append(node)
    
    def build(self) -> SIRModule:
        """Build the SIR module."""
        module = SIRModule(name=self.name)
        
        # Add params
        for name, param in self.params.items():
            module.add_param(name, param)
        
        # Create a main function with all ops
        block = SIRBlock(nodes=self.ops, result=self.outputs[0] if self.outputs else None)
        
        func_params = [(name, props) for name, props in self.inputs]
        func = SIRFunction(
            name="forward",
            params=func_params,
            body=block,
            return_type=FloatType()
        )
        module.add_function(func)
        
        return module
