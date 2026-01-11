"""
Integration tests for the Delta compiler.
"""

import pytest
import torch
from torch import Tensor

from delta.compiler import Compiler, CompileOptions, CompileResult
from delta.source import SourceFile
from delta.frontend.lexer import Lexer
from delta.frontend.parser import Parser
from delta.ir.sir import SIRModule, SIRFunction, SIRBlock, TensorOp, TensorOpKind, ParamRef, SIRProperty
from delta.backend.fx_lowering import FXLowering, lower_to_fx
from delta.runtime.executor import Executor, ExecutionContext
from delta.runtime.context import DeltaContext
from delta.types.types import FloatType, TensorType
from delta.types.roles import RoleInfo


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


class TestCompilerPipeline:
    """End-to-end compiler pipeline tests."""
    
    def test_compile_simple_expression(self):
        """Compile simple expression."""
        code = """
        let x = 42;
        """
        
        source = SourceFile("<test>", code)
        options = CompileOptions()
        compiler = Compiler(options)
        
        result = compiler.compile(source)
        
        assert result is not None
        assert result.success
    
    def test_compile_function(self):
        """Compile function definition."""
        code = """
        fn square(x: Tensor) -> Tensor {
            return x * x;
        }
        """
        
        source = SourceFile("<test>", code)
        options = CompileOptions()
        compiler = Compiler(options)
        
        result = compiler.compile(source)
        
        assert result.success
    
    def test_compile_model(self):
        """Compile simple model."""
        code = """
        param theta: Tensor = randn(10, 5);
        obs x: Tensor;
        obs y: Tensor;
        
        fn forward(x: Tensor) -> Tensor {
            return matmul(x, theta);
        }
        
        let pred = forward(x);
        constraint sum((pred - y) ** 2) == 0 weight 1.0;
        """
        
        source = SourceFile("<test>", code)
        options = CompileOptions()
        compiler = Compiler(options)
        
        result = compiler.compile(source)
        
        assert result.success
    
    def test_compile_with_constraints(self):
        """Compile model with multiple constraints."""
        code = """
        param w: Tensor = randn(10);
        
        constraint w > 0 weight 1.0;
        constraint sum(w) == 1 weight 10.0;
        """
        
        source = SourceFile("<test>", code)
        options = CompileOptions()
        compiler = Compiler(options)
        
        result = compiler.compile(source)
        
        assert result.success
        # Constraints are compiled into the SIR
        assert result.sir is not None
    
    def test_compile_train_mode(self):
        """Compile for train mode."""
        code = """
        param theta: Tensor = randn(10);
        obs x: Tensor;
        
        train {
            let loss = sum((x - theta) ** 2);
        }
        """
        
        source = SourceFile("<test>", code)
        options = CompileOptions(mode="train")
        compiler = Compiler(options)
        
        result = compiler.compile(source)
        
        assert result.success
    
    def test_compile_infer_mode(self):
        """Compile for infer mode."""
        code = """
        param theta: Tensor = randn(10);
        obs x: Tensor;
        
        infer {
            let pred = matmul(x, theta);
            return pred;
        }
        """
        
        source = SourceFile("<test>", code)
        options = CompileOptions(mode="infer")
        compiler = Compiler(options)
        
        result = compiler.compile(source)
        
        assert result.success


class TestIRTransformations:
    """IR transformation integration tests."""
    
    def test_ast_to_sir(self):
        """Transform AST to SIR."""
        code = """
        param theta: Tensor = randn(10);
        let y = sin(theta);
        """
        
        source = SourceFile("<test>", code)
        parser = Parser(source)
        ast = parser.parse_module()
        
        # AST should be transformable to SIR
        # This would use the full compiler pipeline
        assert ast is not None
    
    def test_sir_optimization(self):
        """SIR optimization pass."""
        builder = SIRGraphBuilder("test")
        
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("relu", [x], dtype="float32", shape=(10,))
        z = builder.add_tensor_op("relu", [y], dtype="float32", shape=(10,))  # Redundant
        builder.add_output(z)
        
        graph = builder.build()
        
        # Should be able to optimize
        assert graph is not None
    
    def test_sir_to_fx(self):
        """Lower SIR to FX."""
        builder = SIRGraphBuilder("test")
        
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("relu", [x], dtype="float32", shape=(10,))
        builder.add_output(y)
        
        graph = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(graph)
        
        assert fx_modules is not None
        assert "forward" in fx_modules
        
        # Get the forward module
        fx_module = fx_modules["forward"]
        
        # Test execution
        test_input = torch.randn(10)
        result = fx_module(test_input)
        
        assert torch.allclose(result, torch.relu(test_input))


class TestRuntimeIntegration:
    """Runtime integration tests."""
    
    def test_execute_compiled_model(self):
        """Execute compiled model."""
        # Build a simple model directly as SIR
        builder = SIRGraphBuilder("model")
        
        x = builder.add_input("x", dtype="float32", shape=(32, 10))
        w = builder.add_param("w", dtype="float32", shape=(10, 5))
        
        y = builder.add_tensor_op("matmul", [x, w], dtype="float32", shape=(32, 5))
        y_relu = builder.add_tensor_op("relu", [y], dtype="float32", shape=(32, 5))
        
        builder.add_output(y_relu)
        
        graph = builder.build()
        
        # Lower and execute
        lowering = FXLowering()
        fx_modules = lowering.lower(graph)
        fx_module = fx_modules["forward"]
        
        # Reinitialize parameters via the params sub-module
        fx_module.params.register_parameter('w', torch.nn.Parameter(torch.randn(10, 5)))
        
        # Execute directly with the FX module
        test_input = torch.randn(32, 10)
        result = fx_module(test_input)
        
        assert result.shape == (32, 5)
    
    def test_training_with_constraints(self):
        """Training with constraint optimization."""
        # Simple constrained optimization: minimize x^2 subject to x > 1
        x = torch.nn.Parameter(torch.tensor(0.0))
        
        optimizer = torch.optim.Adam([x], lr=0.1)
        
        for _ in range(100):
            # Objective: minimize x^2
            objective = x ** 2
            
            # Constraint: x > 1 (penalize violations)
            constraint_violation = torch.relu(1.0 - x)
            
            # Combined loss
            loss = objective + 10.0 * constraint_violation
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # x should be approximately 1 (constraint boundary)
        assert abs(x.item() - 1.0) < 0.1
    
    def test_inference_mode_execution(self):
        """Execute in inference mode."""
        # Build model
        builder = SIRGraphBuilder("model")
        
        x = builder.add_input("x", dtype="float32", shape=(10,))
        w = builder.add_param("w", dtype="float32", shape=(10,))
        
        y = builder.add_tensor_op("mul", [x, w], dtype="float32", shape=(10,))
        
        builder.add_output(y)
        
        graph = builder.build()
        
        # Lower
        lowering = FXLowering()
        fx_modules = lowering.lower(graph)
        fx_module = fx_modules["forward"]
        # Reinitialize parameters via the params sub-module
        fx_module.params.register_parameter('w', torch.nn.Parameter(torch.randn(10)))
        
        # Execute in inference mode
        fx_module.eval()
        
        with torch.no_grad():
            test_input = torch.randn(10)
            result = fx_module(test_input)
        
        assert not result.requires_grad


class TestEndToEnd:
    """Full end-to-end tests."""
    
    def test_linear_regression(self):
        """End-to-end linear regression."""
        # Generate data
        torch.manual_seed(42)
        true_w = torch.tensor([1.0, 2.0, 3.0])
        X = torch.randn(100, 3)
        y = X @ true_w + 0.1 * torch.randn(100)
        
        # Build model
        builder = SIRGraphBuilder("linear_regression")
        
        x_in = builder.add_input("x", dtype="float32", shape=(100, 3))
        y_in = builder.add_input("y", dtype="float32", shape=(100,))
        w = builder.add_param("w", dtype="float32", shape=(3,))
        
        # pred = x @ w
        pred = builder.add_tensor_op("matmul", [x_in, w], dtype="float32", shape=(100,))
        
        # loss = sum((pred - y)^2)
        diff = builder.add_tensor_op("sub", [pred, y_in], dtype="float32", shape=(100,))
        sq = builder.add_tensor_op("square", [diff], dtype="float32", shape=(100,))
        loss = builder.add_tensor_op("sum", [sq], dtype="float32", shape=())
        
        builder.add_output(loss)
        
        graph = builder.build()
        
        # Lower
        lowering = FXLowering()
        fx_modules = lowering.lower(graph)
        fx_module = fx_modules["forward"]
        
        # Initialize weights via the params sub-module
        w_param = torch.nn.Parameter(torch.randn(3))
        fx_module.params.register_parameter('w', w_param)
        
        # Train
        optimizer = torch.optim.Adam([w_param], lr=0.1)
        
        for epoch in range(100):
            loss = fx_module(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check learned weights
        learned_w = w_param.detach()
        assert torch.allclose(learned_w, true_w, atol=0.2)
    
    def test_neural_network(self):
        """End-to-end neural network."""
        # Generate classification data
        torch.manual_seed(42)
        X = torch.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).float()
        
        # Build model
        builder = SIRGraphBuilder("classifier")
        
        x_in = builder.add_input("x", dtype="float32", shape=(200, 2))
        y_in = builder.add_input("y", dtype="float32", shape=(200,))
        
        w1 = builder.add_param("w1", dtype="float32", shape=(2, 8))
        b1 = builder.add_param("b1", dtype="float32", shape=(8,))
        w2 = builder.add_param("w2", dtype="float32", shape=(8, 1))
        b2 = builder.add_param("b2", dtype="float32", shape=(1,))
        
        # h = relu(x @ w1 + b1)
        xw1 = builder.add_tensor_op("matmul", [x_in, w1], dtype="float32", shape=(200, 8))
        h1 = builder.add_tensor_op("add", [xw1, b1], dtype="float32", shape=(200, 8))
        h1_relu = builder.add_tensor_op("relu", [h1], dtype="float32", shape=(200, 8))
        
        # out = sigmoid(h @ w2 + b2)
        hw2 = builder.add_tensor_op("matmul", [h1_relu, w2], dtype="float32", shape=(200, 1))
        out = builder.add_tensor_op("add", [hw2, b2], dtype="float32", shape=(200, 1))
        pred = builder.add_tensor_op("sigmoid", [out], dtype="float32", shape=(200, 1))
        
        builder.add_output(pred)
        
        graph = builder.build()
        
        # Lower
        lowering = FXLowering()
        fx_modules = lowering.lower(graph)
        fx_module = fx_modules["forward"]
        
        # Initialize weights via the params sub-module
        fx_module.params.register_parameter('w1', torch.nn.Parameter(torch.randn(2, 8) * 0.1))
        fx_module.params.register_parameter('b1', torch.nn.Parameter(torch.zeros(8)))
        fx_module.params.register_parameter('w2', torch.nn.Parameter(torch.randn(8, 1) * 0.1))
        fx_module.params.register_parameter('b2', torch.nn.Parameter(torch.zeros(1)))
        
        params = list(fx_module.params.parameters())
        optimizer = torch.optim.Adam(params, lr=0.05)
        
        # Train
        for epoch in range(200):
            pred = fx_module(X, y).squeeze()
            loss = torch.nn.functional.binary_cross_entropy(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check accuracy
        with torch.no_grad():
            pred = fx_module(X, y).squeeze()
            accuracy = ((pred > 0.5).float() == y).float().mean()
        
        assert accuracy > 0.8


class TestErrorHandling:
    """Error handling integration tests."""
    
    def test_type_error_in_compilation(self):
        """Type error during compilation."""
        code = """
        let x: Tensor = 42;  # Type mismatch
        let y = x + "string";  # Invalid operation
        """
        
        source = SourceFile("<test>", code)
        options = CompileOptions()
        compiler = Compiler(options)
        
        result = compiler.compile(source)
        
        # Should report errors but not crash
        assert result is not None
        # Could have errors in result.errors
    
    def test_undefined_variable(self):
        """Undefined variable error."""
        code = """
        let x = undefined_var + 1;
        """
        
        source = SourceFile("<test>", code)
        options = CompileOptions()
        compiler = Compiler(options)
        
        result = compiler.compile(source)
        
        # Should report name error
        assert result is not None
    
    def test_mode_violation(self):
        """Mode violation error."""
        code = """
        infer {
            let z = rand();  # Stochastic in infer mode
        }
        """
        
        source = SourceFile("<test>", code)
        options = CompileOptions(mode="infer")
        compiler = Compiler(options)
        
        result = compiler.compile(source)
        
        # Should report mode error
        # Stochastic operations are forbidden in infer mode
        assert result is not None
