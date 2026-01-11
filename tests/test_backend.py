"""
Tests for the PyTorch FX backend.

Tests FX lowering from SIR to executable PyTorch modules.
"""

import pytest
import torch
from torch import fx

from delta.ir.sir import (
    SIRModule, SIRFunction, SIRBlock, SIRProperty,
    TensorOp, TensorOpKind, GateOp, MixOp, ConstraintOp, ConstraintKind,
    ParamRef, ObsRef, Const, StopGrad, Harden,
    BinaryTensorOp, UnaryTensorOp, ReduceOp, RoleInfo
)
from delta.ir.sir_builder import SIRGraphBuilder
from delta.backend.fx_lowering import FXLowering
from delta.types.types import FloatType


class TestFXLoweringBasic:
    """Basic FX lowering tests."""
    
    def test_lower_empty_module(self):
        """Lower an empty module."""
        module = SIRModule(name="empty")
        
        # Add a minimal function
        block = SIRBlock(nodes=[], result=None)
        func = SIRFunction(
            name="forward",
            params=[],
            body=block,
            return_type=FloatType()
        )
        module.add_function(func)
        
        lowering = FXLowering()
        result = lowering.lower(module)
        
        assert result is not None
        assert "forward" in result
    
    def test_lower_identity_function(self):
        """Lower an identity function."""
        builder = SIRGraphBuilder("identity")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        builder.add_output(x)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        # Test execution
        test_input = torch.randn(10)
        result = fx_module(test_input)
        
        assert torch.allclose(result, test_input)


class TestFXLoweringTensorOps:
    """Tests for tensor operation lowering."""
    
    def test_lower_relu(self):
        """Lower ReLU operation."""
        builder = SIRGraphBuilder("relu_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("relu", [x], dtype="float32", shape=(10,))
        builder.add_output(y)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        # Test with mixed positive/negative input
        test_input = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, -2.0, 3.0, -3.0, 4.0])
        result = fx_module(test_input)
        expected = torch.relu(test_input)
        
        assert torch.allclose(result, expected)
    
    def test_lower_sigmoid(self):
        """Lower sigmoid operation."""
        builder = SIRGraphBuilder("sigmoid_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("sigmoid", [x], dtype="float32", shape=(10,))
        builder.add_output(y)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        test_input = torch.randn(10)
        result = fx_module(test_input)
        expected = torch.sigmoid(test_input)
        
        assert torch.allclose(result, expected)
    
    def test_lower_tanh(self):
        """Lower tanh operation."""
        builder = SIRGraphBuilder("tanh_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("tanh", [x], dtype="float32", shape=(10,))
        builder.add_output(y)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        test_input = torch.randn(10)
        result = fx_module(test_input)
        expected = torch.tanh(test_input)
        
        assert torch.allclose(result, expected)
    
    def test_lower_exp(self):
        """Lower exp operation."""
        builder = SIRGraphBuilder("exp_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("exp", [x], dtype="float32", shape=(10,))
        builder.add_output(y)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        test_input = torch.randn(10)
        result = fx_module(test_input)
        expected = torch.exp(test_input)
        
        assert torch.allclose(result, expected)
    
    def test_lower_log(self):
        """Lower log operation."""
        builder = SIRGraphBuilder("log_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("log", [x], dtype="float32", shape=(10,))
        builder.add_output(y)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        test_input = torch.rand(10) + 0.1  # Positive values
        result = fx_module(test_input)
        expected = torch.log(test_input)
        
        assert torch.allclose(result, expected)


class TestFXLoweringBinaryOps:
    """Tests for binary operation lowering."""
    
    def test_lower_add(self):
        """Lower add operation."""
        builder = SIRGraphBuilder("add_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_input("y", dtype="float32", shape=(10,))
        z = builder.add_tensor_op("add", [x, y], dtype="float32", shape=(10,))
        builder.add_output(z)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        a = torch.randn(10)
        b = torch.randn(10)
        result = fx_module(a, b)
        expected = a + b
        
        assert torch.allclose(result, expected)
    
    def test_lower_sub(self):
        """Lower sub operation."""
        builder = SIRGraphBuilder("sub_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_input("y", dtype="float32", shape=(10,))
        z = builder.add_tensor_op("sub", [x, y], dtype="float32", shape=(10,))
        builder.add_output(z)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        a = torch.randn(10)
        b = torch.randn(10)
        result = fx_module(a, b)
        expected = a - b
        
        assert torch.allclose(result, expected)
    
    def test_lower_mul(self):
        """Lower mul operation."""
        builder = SIRGraphBuilder("mul_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_input("y", dtype="float32", shape=(10,))
        z = builder.add_tensor_op("mul", [x, y], dtype="float32", shape=(10,))
        builder.add_output(z)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        a = torch.randn(10)
        b = torch.randn(10)
        result = fx_module(a, b)
        expected = a * b
        
        assert torch.allclose(result, expected)
    
    def test_lower_div(self):
        """Lower div operation."""
        builder = SIRGraphBuilder("div_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_input("y", dtype="float32", shape=(10,))
        z = builder.add_tensor_op("div", [x, y], dtype="float32", shape=(10,))
        builder.add_output(z)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        a = torch.randn(10)
        b = torch.randn(10) + 1.0  # Avoid division by zero
        result = fx_module(a, b)
        expected = a / b
        
        assert torch.allclose(result, expected)
    
    def test_lower_matmul(self):
        """Lower matmul operation."""
        builder = SIRGraphBuilder("matmul_model")
        x = builder.add_input("x", dtype="float32", shape=(10, 5))
        y = builder.add_input("y", dtype="float32", shape=(5, 3))
        z = builder.add_tensor_op("matmul", [x, y], dtype="float32", shape=(10, 3))
        builder.add_output(z)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        a = torch.randn(10, 5)
        b = torch.randn(5, 3)
        result = fx_module(a, b)
        expected = torch.matmul(a, b)
        
        assert torch.allclose(result, expected)


class TestFXLoweringReductions:
    """Tests for reduction operation lowering."""
    
    def test_lower_sum(self):
        """Lower sum operation."""
        builder = SIRGraphBuilder("sum_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("sum", [x], dtype="float32", shape=())
        builder.add_output(y)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        test_input = torch.randn(10)
        result = fx_module(test_input)
        expected = test_input.sum()
        
        assert torch.allclose(result, expected)
    
    def test_lower_mean(self):
        """Lower mean operation."""
        builder = SIRGraphBuilder("mean_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("mean", [x], dtype="float32", shape=())
        builder.add_output(y)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        test_input = torch.randn(10)
        result = fx_module(test_input)
        expected = test_input.mean()
        
        assert torch.allclose(result, expected)


class TestFXLoweringWithParams:
    """Tests for lowering with parameters."""
    
    def test_lower_with_param(self):
        """Lower model with learnable parameter."""
        builder = SIRGraphBuilder("param_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        w = builder.add_param("w", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("mul", [x, w], dtype="float32", shape=(10,))
        builder.add_output(y)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        # Initialize parameter
        fx_module.params.register_parameter("w", torch.nn.Parameter(torch.ones(10)))
        
        test_input = torch.randn(10)
        result = fx_module(test_input)
        
        # With w = ones, output should equal input
        assert torch.allclose(result, test_input)
    
    def test_lower_with_multiple_params(self):
        """Lower model with multiple parameters."""
        builder = SIRGraphBuilder("multi_param_model")
        x = builder.add_input("x", dtype="float32", shape=(5, 3))
        w = builder.add_param("w", dtype="float32", shape=(3, 2))
        b = builder.add_param("b", dtype="float32", shape=(2,))
        
        xw = builder.add_tensor_op("matmul", [x, w], dtype="float32", shape=(5, 2))
        y = builder.add_tensor_op("add", [xw, b], dtype="float32", shape=(5, 2))
        builder.add_output(y)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        # Initialize parameters
        fx_module.params.register_parameter("w", torch.nn.Parameter(torch.randn(3, 2)))
        fx_module.params.register_parameter("b", torch.nn.Parameter(torch.zeros(2)))
        
        test_input = torch.randn(5, 3)
        result = fx_module(test_input)
        
        assert result.shape == (5, 2)


class TestFXLoweringComposite:
    """Tests for composite operations."""
    
    def test_lower_mlp_layer(self):
        """Lower a simple MLP layer: y = relu(x @ w + b)."""
        builder = SIRGraphBuilder("mlp_layer")
        x = builder.add_input("x", dtype="float32", shape=(32, 10))
        w = builder.add_param("w", dtype="float32", shape=(10, 5))
        b = builder.add_param("b", dtype="float32", shape=(5,))
        
        xw = builder.add_tensor_op("matmul", [x, w], dtype="float32", shape=(32, 5))
        xwb = builder.add_tensor_op("add", [xw, b], dtype="float32", shape=(32, 5))
        y = builder.add_tensor_op("relu", [xwb], dtype="float32", shape=(32, 5))
        builder.add_output(y)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        # Initialize parameters
        fx_module.params.register_parameter("w", torch.nn.Parameter(torch.randn(10, 5) * 0.1))
        fx_module.params.register_parameter("b", torch.nn.Parameter(torch.zeros(5)))
        
        test_input = torch.randn(32, 10)
        result = fx_module(test_input)
        
        assert result.shape == (32, 5)
        # Result should be non-negative due to ReLU
        assert (result >= 0).all()


class TestFXLoweringGradients:
    """Tests for gradient computation."""
    
    def test_gradients_flow(self):
        """Verify gradients flow through lowered model."""
        builder = SIRGraphBuilder("grad_model")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        w = builder.add_param("w", dtype="float32", shape=(10,))
        
        prod = builder.add_tensor_op("mul", [x, w], dtype="float32", shape=(10,))
        y = builder.add_tensor_op("sum", [prod], dtype="float32", shape=())
        builder.add_output(y)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        # Initialize parameter with gradient tracking
        w_param = torch.nn.Parameter(torch.ones(10))
        fx_module.params.register_parameter("w", w_param)
        
        test_input = torch.randn(10)
        result = fx_module(test_input)
        
        # Backpropagate
        result.backward()
        
        # Gradient should equal input (d(sum(x*w))/dw = x)
        assert w_param.grad is not None
        assert torch.allclose(w_param.grad, test_input)


class TestFXModuleExecution:
    """Tests for FX module execution."""
    
    def test_module_is_callable(self):
        """Lowered module is callable."""
        builder = SIRGraphBuilder("callable")
        x = builder.add_input("x", dtype="float32", shape=(5,))
        builder.add_output(x)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        assert callable(fx_module)
    
    def test_module_eval_mode(self):
        """Module can be put in eval mode."""
        builder = SIRGraphBuilder("eval_mode")
        x = builder.add_input("x", dtype="float32", shape=(5,))
        builder.add_output(x)
        
        module = builder.build()
        
        lowering = FXLowering()
        fx_modules = lowering.lower(module)
        fx_module = fx_modules["forward"]
        
        fx_module.eval()
        assert not fx_module.training
        
        fx_module.train()
        assert fx_module.training
