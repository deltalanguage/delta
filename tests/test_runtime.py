"""
Tests for Delta runtime.

Tests context management, execution modes, and runtime utilities.
"""

import pytest
import torch

from delta.runtime.context import DeltaContext


class TestDeltaContext:
    """Tests for DeltaContext."""
    
    def test_default_context(self):
        """Default context has sensible defaults."""
        ctx = DeltaContext()
        
        assert ctx.mode == "train"
        assert ctx.device == torch.device("cpu")
        assert ctx.dtype == torch.float32
        assert ctx.debug is False
    
    def test_context_with_seed(self):
        """Context can set random seed."""
        ctx = DeltaContext(seed=42)
        
        assert ctx.seed == 42
        # Seed should affect torch random state
        a = torch.randn(10)
        
        ctx.set_seed(42)
        b = torch.randn(10)
        
        assert torch.allclose(a, b)
    
    def test_to_device(self):
        """to_device moves tensors to context device."""
        ctx = DeltaContext(dtype=torch.float64)
        
        tensor = torch.tensor([1.0, 2.0, 3.0])
        moved = ctx.to_device(tensor)
        
        assert moved.dtype == torch.float64


class TestContextModes:
    """Tests for context mode switching."""
    
    def test_train_mode_context(self):
        """train_mode enables gradients."""
        ctx = DeltaContext(mode="infer")
        
        with ctx.train_mode():
            assert ctx.mode == "train"
            # Gradients should be enabled
            x = torch.randn(10, requires_grad=True)
            y = x.sum()
            y.backward()
            assert x.grad is not None
        
        assert ctx.mode == "infer"  # Restored
    
    def test_infer_mode_context(self):
        """infer_mode disables gradients."""
        ctx = DeltaContext(mode="train")
        
        with ctx.infer_mode():
            assert ctx.mode == "infer"
            # Operations should not track gradients
            x = torch.randn(10, requires_grad=True)
            y = x.sum()
            # Can't backward in no_grad context
        
        assert ctx.mode == "train"  # Restored
    
    def test_analyze_mode_context(self):
        """analyze_mode sets mode correctly."""
        ctx = DeltaContext()
        
        with ctx.analyze_mode():
            assert ctx.mode == "analyze"
        
        assert ctx.mode == "train"


class TestContextSeeding:
    """Tests for random seed management."""
    
    def test_reproducible_random(self):
        """Same seed produces same random values."""
        ctx1 = DeltaContext(seed=123)
        a1 = torch.randn(100)
        
        ctx2 = DeltaContext(seed=123)
        a2 = torch.randn(100)
        
        assert torch.allclose(a1, a2)
    
    def test_different_seeds(self):
        """Different seeds produce different random values."""
        ctx1 = DeltaContext(seed=123)
        a1 = torch.randn(100)
        
        ctx2 = DeltaContext(seed=456)
        a2 = torch.randn(100)
        
        assert not torch.allclose(a1, a2)


class TestContextDevice:
    """Tests for device handling."""
    
    def test_cpu_device(self):
        """CPU device is default."""
        ctx = DeltaContext()
        assert ctx.device == torch.device("cpu")
        
    def test_set_device(self):
        """Can set device explicitly."""
        ctx = DeltaContext(device=torch.device("cpu"))
        assert ctx.device.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Can use CUDA device if available."""
        ctx = DeltaContext(device=torch.device("cuda"))
        
        tensor = torch.randn(10)
        moved = ctx.to_device(tensor)
        
        assert moved.device.type == "cuda"


class TestContextDebug:
    """Tests for debug functionality."""
    
    def test_debug_off_by_default(self):
        """Debug is off by default."""
        ctx = DeltaContext()
        assert ctx.debug is False
    
    def test_enable_debug(self):
        """Can enable debug mode."""
        ctx = DeltaContext(debug=True)
        assert ctx.debug is True
    
    def test_trace_flags(self):
        """Can set trace flags."""
        ctx = DeltaContext(
            trace_gradients=True,
            trace_activations=True
        )
        
        assert ctx.trace_gradients is True
        assert ctx.trace_activations is True


class TestContextDtype:
    """Tests for dtype handling."""
    
    def test_default_dtype(self):
        """Default dtype is float32."""
        ctx = DeltaContext()
        assert ctx.dtype == torch.float32
    
    def test_float64_dtype(self):
        """Can use float64."""
        ctx = DeltaContext(dtype=torch.float64)
        
        tensor = torch.tensor([1.0, 2.0])
        moved = ctx.to_device(tensor)
        
        assert moved.dtype == torch.float64
    
    def test_float16_dtype(self):
        """Can use float16."""
        ctx = DeltaContext(dtype=torch.float16)
        
        tensor = torch.tensor([1.0, 2.0])
        moved = ctx.to_device(tensor)
        
        assert moved.dtype == torch.float16
