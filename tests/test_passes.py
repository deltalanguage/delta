"""
Tests for Delta compiler passes.

Tests individual compiler passes and their transformations.
"""

import pytest
from delta.ir.sir import (
    SIRModule, SIRFunction, SIRBlock, SIRProperty,
    TensorOp, TensorOpKind, GateOp, MixOp, ConstraintOp, ConstraintKind,
    ParamRef, ObsRef, Const, StopGrad, Harden, ModeSwitch,
    BinaryTensorOp, UnaryTensorOp, RoleInfo
)
from delta.ir.sir_builder import SIRGraphBuilder
from delta.types.types import FloatType, BoolType
from delta.passes.relaxation import RelaxationPass, RelaxationConfig


class TestRelaxationPass:
    """Tests for the relaxation pass."""
    
    def test_relaxation_preserves_module_structure(self):
        """Relaxation preserves module name and function names."""
        builder = SIRGraphBuilder("test_module")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("relu", [x], dtype="float32", shape=(10,))
        builder.add_output(y)
        
        module = builder.build()
        
        relaxation = RelaxationPass()
        relaxed = relaxation.relax(module)
        
        assert relaxed.name == module.name
        assert "forward" in relaxed.functions
    
    def test_relaxation_preserves_params(self):
        """Relaxation preserves parameter references."""
        builder = SIRGraphBuilder("param_test")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        w = builder.add_param("w", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("mul", [x, w], dtype="float32", shape=(10,))
        builder.add_output(y)
        
        module = builder.build()
        
        relaxation = RelaxationPass()
        relaxed = relaxation.relax(module)
        
        assert "w" in relaxed.params
    
    def test_relaxation_config_defaults(self):
        """RelaxationConfig has sensible defaults."""
        config = RelaxationConfig()
        
        assert config.default_temperature == 1.0
        assert config.min_temperature == 0.01
        assert config.anneal_temperature is False
    
    def test_relaxation_custom_config(self):
        """Relaxation uses custom configuration."""
        config = RelaxationConfig(
            default_temperature=0.5,
            anneal_temperature=True
        )
        relaxation = RelaxationPass(config)
        
        assert relaxation.config.default_temperature == 0.5
        assert relaxation.config.anneal_temperature is True


class TestSIRBuilderForPasses:
    """Tests that SIRGraphBuilder creates valid input for passes."""
    
    def test_builder_creates_valid_module(self):
        """Builder creates a valid SIR module."""
        builder = SIRGraphBuilder("valid_module")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        builder.add_output(x)
        
        module = builder.build()
        
        assert isinstance(module, SIRModule)
        assert module.name == "valid_module"
        assert "forward" in module.functions
    
    def test_builder_with_multiple_inputs(self):
        """Builder handles multiple inputs."""
        builder = SIRGraphBuilder("multi_input")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_input("y", dtype="float32", shape=(10,))
        z = builder.add_tensor_op("add", [x, y], dtype="float32", shape=(10,))
        builder.add_output(z)
        
        module = builder.build()
        func = module.functions["forward"]
        
        assert len(func.params) == 2
    
    def test_builder_with_params(self):
        """Builder handles learnable parameters."""
        builder = SIRGraphBuilder("with_params")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        w = builder.add_param("weight", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("mul", [x, w], dtype="float32", shape=(10,))
        builder.add_output(y)
        
        module = builder.build()
        
        assert "weight" in module.params


class TestPassComposition:
    """Tests for composing multiple passes."""
    
    def test_relaxation_idempotent(self):
        """Relaxation applied twice gives same result."""
        builder = SIRGraphBuilder("idempotent_test")
        x = builder.add_input("x", dtype="float32", shape=(10,))
        y = builder.add_tensor_op("relu", [x], dtype="float32", shape=(10,))
        builder.add_output(y)
        
        module = builder.build()
        
        relaxation = RelaxationPass()
        relaxed1 = relaxation.relax(module)
        relaxed2 = relaxation.relax(relaxed1)
        
        # Structure should be preserved
        assert relaxed2.name == relaxed1.name
        assert len(relaxed2.functions) == len(relaxed1.functions)


class TestModeSwitch:
    """Tests for mode-specific transformations."""
    
    def test_mode_switch_creation(self):
        """ModeSwitch can be created."""
        train_val = Const(value=1.0)
        infer_val = Const(value=0.0)
        
        switch = ModeSwitch(
            train_value=train_val,
            infer_value=infer_val
        )
        
        assert switch.train_value is train_val
        assert switch.infer_value is infer_val


class TestGateOp:
    """Tests for gate operations in passes."""
    
    def test_gate_op_creation(self):
        """GateOp can be created for soft comparisons."""
        lhs = Const(value=1.0)
        rhs = Const(value=2.0)
        temp = Const(value=0.1)
        
        gate = GateOp(
            compare=TensorOpKind.GT,
            lhs=lhs,
            rhs=rhs,
            temperature=temp
        )
        
        assert gate.compare == TensorOpKind.GT
        assert len(gate.inputs()) == 3


class TestMixOp:
    """Tests for mix operations in passes."""
    
    def test_mix_op_creation(self):
        """MixOp can be created for soft conditionals."""
        gate = Const(value=0.7)
        then_val = Const(value=1.0)
        else_val = Const(value=0.0)
        
        mix = MixOp(
            gate=gate,
            then_value=then_val,
            else_value=else_val
        )
        
        assert mix.gate is gate
        assert mix.then_value is then_val
        assert mix.else_value is else_val
