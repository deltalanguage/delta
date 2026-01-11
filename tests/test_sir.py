"""
Tests for the Semantic IR (SIR).

Tests SIR node creation, properties, graph structure, and utilities.
"""

import pytest
from delta.ir.sir import (
    SIRNode, SIRModule, SIRFunction, SIRBlock, SIRProperty,
    TensorOp, TensorOpKind, GateOp, MixOp, ConstraintOp, ConstraintKind,
    ParamRef, ObsRef, Const, StopGrad, Harden, RandomVar, Observe,
    GradBoundary, ModeSwitch, RoleInfo, Mode,
    BinaryTensorOp, UnaryTensorOp, ReduceOp,
    walk_sir, replace_inputs, fresh_node_id
)
from delta.types.types import FloatType, TensorType, BoolType
from delta.types.roles import Role


class TestSIRProperty:
    """Tests for SIR node properties."""
    
    def test_default_property(self):
        """Default SIRProperty has sensible defaults."""
        props = SIRProperty()
        
        assert isinstance(props.dtype, FloatType)
        assert props.shape is None
        assert props.requires_grad is False
        assert props.role.primary_role == Role.CONST
    
    def test_property_with_dtype(self):
        """SIRProperty.with_dtype creates a copy."""
        props = SIRProperty()
        new_props = props.with_dtype(BoolType())
        
        assert isinstance(new_props.dtype, BoolType)
        assert isinstance(props.dtype, FloatType)  # Original unchanged
    
    def test_property_with_grad(self):
        """SIRProperty.with_grad creates a copy."""
        props = SIRProperty(requires_grad=False)
        new_props = props.with_grad(True)
        
        assert new_props.requires_grad is True
        assert props.requires_grad is False  # Original unchanged


class TestTensorOp:
    """Tests for tensor operations."""
    
    def test_create_tensor_op(self):
        """Create a basic tensor operation."""
        const = Const(value=1.0)
        op = TensorOp(op=TensorOpKind.RELU, operands=[const])
        
        assert op.op == TensorOpKind.RELU
        assert len(op.inputs()) == 1
        assert op.inputs()[0] is const
    
    def test_tensor_op_has_unique_id(self):
        """Each tensor op has a unique ID."""
        op1 = TensorOp(op=TensorOpKind.ADD, operands=[])
        op2 = TensorOp(op=TensorOpKind.ADD, operands=[])
        
        assert op1.id != op2.id
    
    def test_binary_tensor_op(self):
        """BinaryTensorOp convenience function."""
        a = Const(value=1.0)
        b = Const(value=2.0)
        op = BinaryTensorOp(TensorOpKind.ADD, a, b)
        
        assert op.op == TensorOpKind.ADD
        assert len(op.operands) == 2
    
    def test_unary_tensor_op(self):
        """UnaryTensorOp convenience function."""
        x = Const(value=1.0)
        op = UnaryTensorOp(TensorOpKind.EXP, x)
        
        assert op.op == TensorOpKind.EXP
        assert len(op.operands) == 1
    
    def test_reduce_op(self):
        """ReduceOp convenience function."""
        x = Const(value=[1.0, 2.0, 3.0])
        op = ReduceOp(TensorOpKind.SUM, x, axis=0)
        
        assert op.op == TensorOpKind.SUM
        assert op.attrs.get("axis") == 0
    
    def test_tensor_op_clone(self):
        """Clone a tensor op with new inputs."""
        a = Const(value=1.0)
        b = Const(value=2.0)
        op = BinaryTensorOp(TensorOpKind.ADD, a, b)
        
        c = Const(value=3.0)
        cloned = op.clone_with_inputs([a, c])
        
        assert cloned.id != op.id
        assert cloned.operands[1] is c
        assert op.operands[1] is b  # Original unchanged


class TestGateOp:
    """Tests for gate operations (soft comparisons)."""
    
    def test_create_gate_op(self):
        """Create a gate operation."""
        a = Const(value=1.0)
        b = Const(value=2.0)
        temp = Const(value=0.1)
        
        gate = GateOp(
            compare=TensorOpKind.GT,
            lhs=a,
            rhs=b,
            temperature=temp
        )
        
        assert gate.compare == TensorOpKind.GT
        assert gate.lhs is a
        assert gate.rhs is b
        assert gate.temperature is temp
    
    def test_gate_op_inputs(self):
        """Gate op reports its inputs correctly."""
        a = Const(value=1.0)
        b = Const(value=2.0)
        temp = Const(value=0.1)
        
        gate = GateOp(compare=TensorOpKind.LT, lhs=a, rhs=b, temperature=temp)
        
        inputs = gate.inputs()
        assert len(inputs) == 3
        assert a in inputs
        assert b in inputs
        assert temp in inputs


class TestMixOp:
    """Tests for mix operations (soft conditionals)."""
    
    def test_create_mix_op(self):
        """Create a mix operation."""
        gate = Const(value=0.7)
        a = Const(value=1.0)
        b = Const(value=2.0)
        
        mix = MixOp(gate=gate, then_value=a, else_value=b)
        
        assert mix.gate is gate
        assert mix.then_value is a
        assert mix.else_value is b
    
    def test_mix_op_inputs(self):
        """Mix op reports its inputs correctly."""
        gate = Const(value=0.7)
        a = Const(value=1.0)
        b = Const(value=2.0)
        
        mix = MixOp(gate=gate, then_value=a, else_value=b)
        
        inputs = mix.inputs()
        assert len(inputs) == 3


class TestConstraintOp:
    """Tests for constraint operations."""
    
    def test_create_equality_constraint(self):
        """Create an equality constraint."""
        lhs = Const(value=1.0)
        rhs = Const(value=1.0)
        weight = Const(value=1.0)
        
        constraint = ConstraintOp(
            kind=ConstraintKind.EQUALITY,
            lhs=lhs,
            rhs=rhs,
            weight=weight
        )
        
        assert constraint.kind == ConstraintKind.EQUALITY
        assert constraint.lhs is lhs
        assert constraint.rhs is rhs
    
    def test_create_inequality_constraint(self):
        """Create an inequality constraint."""
        lhs = Const(value=5.0)
        weight = Const(value=1.0)
        
        constraint = ConstraintOp(
            kind=ConstraintKind.INEQUALITY,
            lhs=lhs,
            rhs=None,
            weight=weight
        )
        
        assert constraint.kind == ConstraintKind.INEQUALITY


class TestReferenceNodes:
    """Tests for reference nodes (ParamRef, ObsRef, Const)."""
    
    def test_param_ref(self):
        """ParamRef has correct role and gradient."""
        param = ParamRef(name="theta")
        
        assert param.name == "theta"
        assert param.props.role.primary_role == Role.PARAM
        assert param.requires_grad is True
    
    def test_obs_ref(self):
        """ObsRef has correct role and no gradient."""
        obs = ObsRef(name="x")
        
        assert obs.name == "x"
        assert obs.props.role.primary_role == Role.OBS
        assert obs.requires_grad is False
    
    def test_const(self):
        """Const holds a value."""
        const = Const(value=42)
        
        assert const.value == 42
        assert const.requires_grad is False
    
    def test_reference_nodes_have_no_inputs(self):
        """Reference nodes have no inputs."""
        param = ParamRef(name="p")
        obs = ObsRef(name="x")
        const = Const(value=1)
        
        assert param.inputs() == []
        assert obs.inputs() == []
        assert const.inputs() == []


class TestGradientControl:
    """Tests for gradient control nodes."""
    
    def test_stop_grad(self):
        """StopGrad blocks gradient flow."""
        x = ParamRef(name="x")
        stopped = StopGrad(operand=x)
        
        assert stopped.requires_grad is False
        assert stopped.inputs() == [x]
    
    def test_harden(self):
        """Harden converts to boolean."""
        gate = Const(value=0.8)
        hardened = Harden(operand=gate)
        
        assert isinstance(hardened.dtype, BoolType)
        assert hardened.threshold == 0.5
    
    def test_harden_custom_threshold(self):
        """Harden with custom threshold."""
        gate = Const(value=0.8)
        hardened = Harden(operand=gate, threshold=0.9)
        
        assert hardened.threshold == 0.9
    
    def test_grad_boundary(self):
        """GradBoundary marks gradient boundaries."""
        x = ParamRef(name="x")
        boundary = GradBoundary(operand=x, kind="checkpoint")
        
        assert boundary.kind == "checkpoint"
        assert boundary.inputs() == [x]


class TestModeSwitch:
    """Tests for mode-specific behavior."""
    
    def test_mode_switch_train_infer(self):
        """ModeSwitch has train and infer values."""
        train_val = Const(value="train")
        infer_val = Const(value="infer")
        
        switch = ModeSwitch(train_value=train_val, infer_value=infer_val)
        
        assert switch.train_value is train_val
        assert switch.infer_value is infer_val
    
    def test_mode_switch_inputs(self):
        """ModeSwitch reports inputs correctly."""
        train_val = Const(value="train")
        infer_val = Const(value="infer")
        analyze_val = Const(value="analyze")
        
        switch = ModeSwitch(
            train_value=train_val,
            infer_value=infer_val,
            analyze_value=analyze_val
        )
        
        inputs = switch.inputs()
        assert len(inputs) == 3


class TestSIRBlock:
    """Tests for SIR blocks."""
    
    def test_empty_block(self):
        """Empty block has no nodes."""
        block = SIRBlock(nodes=[], result=None)
        
        assert len(block) == 0
        assert block.result is None
    
    def test_block_with_nodes(self):
        """Block contains nodes."""
        a = Const(value=1.0)
        b = Const(value=2.0)
        add = BinaryTensorOp(TensorOpKind.ADD, a, b)
        
        block = SIRBlock(nodes=[a, b, add], result=add)
        
        assert len(block) == 3
        assert block.result is add
    
    def test_block_iteration(self):
        """Block is iterable."""
        nodes = [Const(value=i) for i in range(5)]
        block = SIRBlock(nodes=nodes)
        
        assert list(block) == nodes


class TestSIRFunction:
    """Tests for SIR functions."""
    
    def test_create_function(self):
        """Create a simple function."""
        x = Const(value=0.0)
        block = SIRBlock(nodes=[x], result=x)
        
        func = SIRFunction(
            name="identity",
            params=[("x", SIRProperty())],
            body=block,
            return_type=FloatType()
        )
        
        assert func.name == "identity"
        assert func.param_names == ["x"]
    
    def test_function_with_multiple_params(self):
        """Function with multiple parameters."""
        block = SIRBlock(nodes=[], result=None)
        
        func = SIRFunction(
            name="multi",
            params=[
                ("a", SIRProperty()),
                ("b", SIRProperty()),
                ("c", SIRProperty())
            ],
            body=block,
            return_type=FloatType()
        )
        
        assert func.param_names == ["a", "b", "c"]


class TestSIRModule:
    """Tests for SIR modules."""
    
    def test_create_empty_module(self):
        """Create an empty module."""
        module = SIRModule(name="test")
        
        assert module.name == "test"
        assert len(module.functions) == 0
        assert len(module.params) == 0
    
    def test_add_function(self):
        """Add a function to a module."""
        module = SIRModule(name="test")
        
        block = SIRBlock(nodes=[], result=None)
        func = SIRFunction(
            name="foo",
            params=[],
            body=block,
            return_type=FloatType()
        )
        
        module.add_function(func)
        
        assert "foo" in module.functions
        assert module.functions["foo"] is func
    
    def test_add_param(self):
        """Add a parameter to a module."""
        module = SIRModule(name="test")
        param = ParamRef(name="theta")
        
        module.add_param("theta", param)
        
        assert "theta" in module.params
        assert module.params["theta"] is param
    
    def test_add_constraint(self):
        """Add a constraint to a module."""
        module = SIRModule(name="test")
        
        lhs = Const(value=1.0)
        weight = Const(value=1.0)
        constraint = ConstraintOp(
            kind=ConstraintKind.INEQUALITY,
            lhs=lhs,
            rhs=None,
            weight=weight
        )
        
        module.add_constraint(constraint)
        
        assert len(module.constraints) == 1
        assert module.constraints[0] is constraint


class TestWalkSIR:
    """Tests for SIR graph traversal."""
    
    def test_walk_single_node(self):
        """Walk a single node."""
        node = Const(value=1.0)
        walked = walk_sir(node)
        
        assert len(walked) == 1
        assert walked[0] is node
    
    def test_walk_chain(self):
        """Walk a chain of nodes."""
        a = Const(value=1.0)
        b = UnaryTensorOp(TensorOpKind.EXP, a)
        c = UnaryTensorOp(TensorOpKind.LOG, b)
        
        walked = walk_sir(c)
        
        # Topological order: a, b, c
        assert len(walked) == 3
        assert walked.index(a) < walked.index(b)
        assert walked.index(b) < walked.index(c)
    
    def test_walk_diamond(self):
        """Walk a diamond-shaped graph."""
        a = Const(value=1.0)
        b = UnaryTensorOp(TensorOpKind.EXP, a)
        c = UnaryTensorOp(TensorOpKind.LOG, a)
        d = BinaryTensorOp(TensorOpKind.ADD, b, c)
        
        walked = walk_sir(d)
        
        # a should come before b and c, which come before d
        assert walked.index(a) < walked.index(b)
        assert walked.index(a) < walked.index(c)
        assert walked.index(b) < walked.index(d)
        assert walked.index(c) < walked.index(d)


class TestReplaceInputs:
    """Tests for input replacement."""
    
    def test_replace_input(self):
        """Replace an input in a node."""
        a = Const(value=1.0)
        b = Const(value=2.0)
        op = BinaryTensorOp(TensorOpKind.ADD, a, b)
        
        c = Const(value=3.0)
        new_op = replace_inputs(op, {b.id: c})
        
        assert new_op.operands[0] is a
        assert new_op.operands[1] is c
    
    def test_no_replacement_returns_same(self):
        """No replacement returns the same node."""
        a = Const(value=1.0)
        b = Const(value=2.0)
        op = BinaryTensorOp(TensorOpKind.ADD, a, b)
        
        same = replace_inputs(op, {})
        
        assert same is op


class TestMode:
    """Tests for execution modes."""
    
    def test_mode_values(self):
        """All modes are defined."""
        assert Mode.TRAIN is not None
        assert Mode.INFER is not None
        assert Mode.ANALYZE is not None
    
    def test_mode_string(self):
        """Mode converts to lowercase string."""
        assert str(Mode.TRAIN) == "train"
        assert str(Mode.INFER) == "infer"
        assert str(Mode.ANALYZE) == "analyze"
