"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
from typing import Generator

from delta.source import SourceFile
from delta.frontend.lexer import Lexer
from delta.frontend.parser import Parser
from delta.ir.sir_builder import SIRGraphBuilder
from delta.ir.sir import Mode
from delta.runtime.context import DeltaContext


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def reset_random_state() -> Generator[None, None, None]:
    """Reset random state before each test."""
    torch.manual_seed(42)
    yield


@pytest.fixture
def source_file() -> SourceFile:
    """Create a test source file."""
    return SourceFile("<test>", "")


@pytest.fixture
def simple_graph() -> 'SIRGraphBuilder':
    """Create a simple SIR graph builder."""
    builder = SIRGraphBuilder("test_graph")
    return builder


@pytest.fixture
def runtime_context() -> DeltaContext:
    """Create a runtime context."""
    return DeltaContext()


@pytest.fixture
def train_context() -> DeltaContext:
    """Create a runtime context in train mode."""
    return DeltaContext(mode="train")


@pytest.fixture
def infer_context() -> DeltaContext:
    """Create a runtime context in infer mode."""
    return DeltaContext(mode="infer")


# ============================================================
# Helper Functions
# ============================================================

def parse_code(code: str):
    """Parse Delta code and return AST."""
    source = SourceFile("<test>", code)
    lexer = Lexer(source)
    tokens = list(lexer.tokenize())
    parser = Parser(tokens, source)
    return parser.parse()


def make_tensor(*shape: int, requires_grad: bool = False) -> torch.Tensor:
    """Create a random tensor with given shape."""
    return torch.randn(*shape, requires_grad=requires_grad)


def assert_tensors_close(
    a: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> None:
    """Assert two tensors are close."""
    assert torch.allclose(a, b, rtol=rtol, atol=atol), \
        f"Tensors not close: max diff = {(a - b).abs().max()}"


def assert_gradients_exist(*tensors: torch.Tensor) -> None:
    """Assert all tensors have gradients."""
    for t in tensors:
        assert t.grad is not None, "Expected gradient to exist"


def assert_no_nans(*tensors: torch.Tensor) -> None:
    """Assert no tensors contain NaN values."""
    for t in tensors:
        assert not torch.isnan(t).any(), "Tensor contains NaN values"


# ============================================================
# Markers
# ============================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


# ============================================================
# Skip Conditions
# ============================================================

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)
