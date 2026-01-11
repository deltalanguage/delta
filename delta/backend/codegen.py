"""
Code generation for Delta.

Generates executable Python/PyTorch code from:
- SIR modules
- FX GraphModules

Supports:
- Standalone Python files
- Importable modules
- Jupyter notebook cells
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TextIO
import io

from delta.ir.sir import (
    SIRModule, SIRFunction, SIRBlock, SIRNode,
    TensorOp, TensorOpKind, GateOp, MixOp, ConstraintOp,
    ParamRef, ObsRef, Const, StopGrad, Harden,
    RandomVar, Observe,
)


@dataclass
class CodeGenConfig:
    """Configuration for code generation."""
    include_imports: bool = True
    include_docstrings: bool = True
    include_type_hints: bool = True
    target_python_version: str = "3.10"
    indent_width: int = 4


class CodeGenerator:
    """
    Generates Python code from Delta SIR.
    
    Produces clean, readable Python that uses PyTorch directly.
    """
    
    def __init__(self, config: Optional[CodeGenConfig] = None) -> None:
        self.config = config or CodeGenConfig()
        self.indent_level = 0
        self.output: io.StringIO = io.StringIO()
        self.node_names: dict[int, str] = {}
        self._name_counter = 0
    
    def generate(self, module: SIRModule) -> str:
        """Generate Python code for a SIR module."""
        self.output = io.StringIO()
        self.node_names = {}
        self._name_counter = 0
        
        # Imports
        if self.config.include_imports:
            self._write_imports()
        
        # Module docstring
        if self.config.include_docstrings:
            self._write_line(f'"""Generated from Delta module: {module.name}"""')
            self._write_line("")
        
        # Generate model class
        self._generate_model_class(module)
        
        return self.output.getvalue()
    
    def _write_imports(self) -> None:
        """Write import statements."""
        self._write_line("import torch")
        self._write_line("import torch.nn as nn")
        self._write_line("import torch.nn.functional as F")
        self._write_line("from typing import Optional, Tuple, Dict, Any")
        self._write_line("")
        self._write_line("")
    
    def _generate_model_class(self, module: SIRModule) -> None:
        """Generate the model class."""
        class_name = self._to_class_name(module.name)
        
        self._write_line(f"class {class_name}(nn.Module):")
        self.indent_level += 1
        
        # Docstring
        if self.config.include_docstrings:
            self._write_line('"""')
            self._write_line(f"Delta-generated model: {module.name}")
            self._write_line('"""')
            self._write_line("")
        
        # __init__
        self._generate_init(module)
        self._write_line("")
        
        # Forward methods for each function
        for name, func in module.functions.items():
            self._generate_forward_method(func)
            self._write_line("")
        
        # Constraints method if any
        if module.constraints:
            self._generate_constraints_method(module)
            self._write_line("")
        
        # Training step
        self._generate_training_step(module)
        
        self.indent_level -= 1
    
    def _generate_init(self, module: SIRModule) -> None:
        """Generate __init__ method."""
        self._write_line("def __init__(self):")
        self.indent_level += 1
        self._write_line("super().__init__()")
        
        # Parameters
        for name, param in module.params.items():
            dtype_str = self._dtype_to_torch(param.dtype)
            shape_str = "1"  # Default shape
            if param.shape:
                shape_str = ", ".join(str(d) for d in param.shape)
            
            self._write_line(f"self.{name} = nn.Parameter(torch.zeros({shape_str}))")
        
        if not module.params:
            self._write_line("pass")
        
        self.indent_level -= 1
    
    def _generate_forward_method(self, func: SIRFunction) -> None:
        """Generate a forward method from a SIR function."""
        # Method signature
        params = ", ".join(f"{name}" for name, _ in func.params)
        if params:
            params = f"self, {params}"
        else:
            params = "self"
        
        self._write_line(f"def {func.name}({params}):")
        self.indent_level += 1
        
        # Reset node names for this function
        self.node_names = {}
        self._name_counter = 0
        
        # Generate body
        result = self._generate_block(func.body)
        
        if result:
            self._write_line(f"return {result}")
        else:
            self._write_line("return None")
        
        self.indent_level -= 1
    
    def _generate_block(self, block: SIRBlock) -> Optional[str]:
        """Generate code for a block."""
        for node in block.nodes:
            self._generate_node(node)
        
        if block.result:
            return self._generate_node(block.result)
        return None
    
    def _generate_node(self, node: SIRNode) -> str:
        """Generate code for a node, returning the variable name."""
        if node.id in self.node_names:
            return self.node_names[node.id]
        
        var_name = self._fresh_name()
        code: str
        
        if isinstance(node, TensorOp):
            code = self._generate_tensor_op(node)
        elif isinstance(node, GateOp):
            code = self._generate_gate_op(node)
        elif isinstance(node, MixOp):
            code = self._generate_mix_op(node)
        elif isinstance(node, ConstraintOp):
            code = self._generate_constraint_op(node)
        elif isinstance(node, ParamRef):
            var_name = f"self.{node.name}"
            self.node_names[node.id] = var_name
            return var_name
        elif isinstance(node, ObsRef):
            # Observations are function parameters
            var_name = node.name
            self.node_names[node.id] = var_name
            return var_name
        elif isinstance(node, Const):
            code = self._generate_const(node)
        elif isinstance(node, StopGrad):
            operand = self._generate_node(node.operand)
            code = f"{operand}.detach()"
        elif isinstance(node, Harden):
            operand = self._generate_node(node.operand)
            code = f"({operand} > {node.threshold}).float()"
        elif isinstance(node, RandomVar):
            code = self._generate_random_var(node)
        elif isinstance(node, Observe):
            code = self._generate_observe(node)
        else:
            code = "None  # Unknown node type"
        
        self._write_line(f"{var_name} = {code}")
        self.node_names[node.id] = var_name
        return var_name
    
    def _generate_tensor_op(self, node: TensorOp) -> str:
        """Generate code for a tensor operation."""
        operands = [self._generate_node(op) for op in node.operands]
        
        # Map to PyTorch function
        op_map = {
            TensorOpKind.ADD: ("torch.add", 2),
            TensorOpKind.SUB: ("torch.sub", 2),
            TensorOpKind.MUL: ("torch.mul", 2),
            TensorOpKind.DIV: ("torch.div", 2),
            TensorOpKind.POW: ("torch.pow", 2),
            TensorOpKind.MATMUL: ("torch.matmul", 2),
            TensorOpKind.NEG: ("torch.neg", 1),
            TensorOpKind.EXP: ("torch.exp", 1),
            TensorOpKind.LOG: ("torch.log", 1),
            TensorOpKind.SIN: ("torch.sin", 1),
            TensorOpKind.COS: ("torch.cos", 1),
            TensorOpKind.TANH: ("torch.tanh", 1),
            TensorOpKind.SIGMOID: ("torch.sigmoid", 1),
            TensorOpKind.RELU: ("F.relu", 1),
            TensorOpKind.SOFTMAX: (".softmax(dim=-1)", 1),
            TensorOpKind.SUM: (".sum()", 1),
            TensorOpKind.MEAN: (".mean()", 1),
            TensorOpKind.MAX: (".max()", 1),
            TensorOpKind.MIN: (".min()", 1),
        }
        
        if node.op in op_map:
            func, arity = op_map[node.op]
            
            if func.startswith("."):
                # Method call
                return f"{operands[0]}{func}"
            else:
                # Function call
                if arity == 1:
                    return f"{func}({operands[0]})"
                else:
                    return f"{func}({', '.join(operands[:arity])})"
        
        # Comparisons
        comp_map = {
            TensorOpKind.LT: "<",
            TensorOpKind.LE: "<=",
            TensorOpKind.GT: ">",
            TensorOpKind.GE: ">=",
            TensorOpKind.EQ: "==",
            TensorOpKind.NE: "!=",
        }
        
        if node.op in comp_map:
            return f"({operands[0]} {comp_map[node.op]} {operands[1]}).float()"
        
        return f"# Unknown op: {node.op.name}"
    
    def _generate_gate_op(self, node: GateOp) -> str:
        """Generate code for a gate operation."""
        lhs = self._generate_node(node.lhs)
        rhs = self._generate_node(node.rhs)
        temp = self._generate_node(node.temperature)
        
        # Gate formula: sigmoid((a - b) / temp) or sigmoid((b - a) / temp)
        if node.compare in (TensorOpKind.LT, TensorOpKind.LE):
            return f"torch.sigmoid(({rhs} - {lhs}) / {temp})"
        else:
            return f"torch.sigmoid(({lhs} - {rhs}) / {temp})"
    
    def _generate_mix_op(self, node: MixOp) -> str:
        """Generate code for a mix operation."""
        gate = self._generate_node(node.gate)
        then_val = self._generate_node(node.then_value)
        else_val = self._generate_node(node.else_value)
        
        return f"{gate} * {then_val} + (1 - {gate}) * {else_val}"
    
    def _generate_constraint_op(self, node: ConstraintOp) -> str:
        """Generate code for a constraint operation."""
        lhs = self._generate_node(node.lhs)
        weight = self._generate_node(node.weight)
        
        if node.rhs:
            rhs = self._generate_node(node.rhs)
            diff = f"({lhs} - {rhs})"
        else:
            diff = lhs
        
        if node.kind.name == "EQUALITY":
            return f"{weight} * ({diff} ** 2).mean()"
        elif node.kind.name == "INEQUALITY":
            return f"{weight} * F.softplus({diff}).mean()"
        else:
            return f"{weight} * (1 - {lhs}).mean()"
    
    def _generate_const(self, node: Const) -> str:
        """Generate code for a constant."""
        value = node.value
        
        if value is None:
            return "None"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return f"torch.tensor({value})"
        elif isinstance(value, list):
            return f"torch.tensor({value})"
        else:
            return f"torch.tensor({repr(value)})"
    
    def _generate_random_var(self, node: RandomVar) -> str:
        """Generate code for a random variable."""
        params = [self._generate_node(p) for p in node.params]
        
        if node.distribution == "Normal":
            if len(params) >= 2:
                return f"{params[0]} + {params[1]} * torch.randn_like({params[0]})"
            else:
                return f"{params[0]} + torch.randn_like({params[0]})"
        elif node.distribution == "Bernoulli":
            return f"torch.bernoulli({params[0]})"
        else:
            return f"torch.randn_like({params[0]})" if params else "torch.randn(1)"
    
    def _generate_observe(self, node: Observe) -> str:
        """Generate code for an observation."""
        value = self._generate_node(node.value)
        params = [self._generate_node(p) for p in node.params]
        
        if node.distribution == "Normal":
            if len(params) >= 2:
                return f"-0.5 * ((({value} - {params[0]}) / {params[1]}) ** 2) - torch.log({params[1]})"
            else:
                return f"-0.5 * ({value} - {params[0]}) ** 2"
        else:
            return "torch.tensor(0.0)"
    
    def _generate_constraints_method(self, module: SIRModule) -> None:
        """Generate the compute_constraints method."""
        self._write_line("def compute_constraints(self, **inputs):")
        self.indent_level += 1
        
        self.node_names = {}
        self._name_counter = 0
        
        self._write_line('"""Compute all constraint penalties."""')
        self._write_line("penalties = []")
        
        for i, constraint in enumerate(module.constraints):
            var = self._generate_node(constraint)
            self._write_line(f"penalties.append({var})")
        
        self._write_line("return sum(penalties) if penalties else torch.tensor(0.0)")
        
        self.indent_level -= 1
    
    def _generate_training_step(self, module: SIRModule) -> None:
        """Generate training step method."""
        self._write_line("def training_step(self, **inputs):")
        self.indent_level += 1
        
        self._write_line('"""Execute one training step."""')
        
        # Call first function as the main forward pass
        if module.functions:
            first_func = next(iter(module.functions.keys()))
            params = list(module.functions[first_func].params)
            args = ", ".join(f"inputs['{name}']" for name, _ in params)
            self._write_line(f"output = self.{first_func}({args})")
        else:
            self._write_line("output = None")
        
        if module.constraints:
            self._write_line("constraint_loss = self.compute_constraints(**inputs)")
        else:
            self._write_line("constraint_loss = torch.tensor(0.0)")
        
        self._write_line("return output, constraint_loss")
        
        self.indent_level -= 1
    
    def _write_line(self, line: str) -> None:
        """Write a line with proper indentation."""
        indent = " " * (self.indent_level * self.config.indent_width)
        self.output.write(f"{indent}{line}\n")
    
    def _fresh_name(self) -> str:
        """Generate a fresh variable name."""
        self._name_counter += 1
        return f"_v{self._name_counter}"
    
    def _to_class_name(self, name: str) -> str:
        """Convert a module name to a class name."""
        # Take the last component and capitalize
        base = name.split("/")[-1].split("\\")[-1]
        base = base.replace(".delta", "").replace("-", "_")
        parts = base.split("_")
        return "".join(p.capitalize() for p in parts) + "Model"
    
    def _dtype_to_torch(self, dtype) -> str:
        """Convert Delta type to PyTorch dtype string."""
        from delta.types.types import FloatType, IntType, BoolType
        
        if isinstance(dtype, FloatType):
            return "torch.float32"
        elif isinstance(dtype, IntType):
            return "torch.int64"
        elif isinstance(dtype, BoolType):
            return "torch.bool"
        else:
            return "torch.float32"


def generate_python(module: SIRModule, config: Optional[CodeGenConfig] = None) -> str:
    """Convenience function to generate Python code."""
    generator = CodeGenerator(config)
    return generator.generate(module)
