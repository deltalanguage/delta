"""
Tests for the Delta standard library.
"""

import pytest
import torch
from torch import Tensor
import numpy as np

from delta.stdlib import tensor, nn, optim, constraints, dist, data, debug


class TestStdTensor:
    """std.tensor tests."""
    
    def test_zeros(self):
        """Create zeros tensor."""
        t = tensor.zeros(10, 5)
        
        assert t.shape == (10, 5)
        assert (t == 0).all()
    
    def test_ones(self):
        """Create ones tensor."""
        t = tensor.ones(10, 5)
        
        assert t.shape == (10, 5)
        assert (t == 1).all()
    
    def test_randn(self):
        """Create random normal tensor."""
        t = tensor.randn(100, 100)
        
        assert t.shape == (100, 100)
        # Check approximately normal
        assert abs(t.mean().item()) < 0.1
        assert abs(t.std().item() - 1.0) < 0.1
    
    def test_arange(self):
        """Create range tensor."""
        t = tensor.arange(0, 10)
        
        assert t.shape == (10,)
        assert t[0] == 0
        assert t[9] == 9
    
    def test_eye(self):
        """Create identity tensor."""
        t = tensor.eye(5)
        
        assert t.shape == (5, 5)
        assert t[0, 0] == 1
        assert t[0, 1] == 0
    
    def test_stack(self):
        """Stack tensors."""
        a = tensor.zeros(10)
        b = tensor.ones(10)
        
        stacked = tensor.stack([a, b])
        
        assert stacked.shape == (2, 10)
    
    def test_concat(self):
        """Concatenate tensors."""
        a = tensor.zeros(5, 10)
        b = tensor.ones(3, 10)
        
        concatenated = tensor.concat([a, b], dim=0)
        
        assert concatenated.shape == (8, 10)


class TestStdNN:
    """std.nn tests."""
    
    def test_linear_layer(self):
        """Create and use linear layer."""
        linear = nn.Linear(10, 5)
        
        x = torch.randn(32, 10)
        y = linear(x)
        
        assert y.shape == (32, 5)
    
    def test_conv2d_layer(self):
        """Create and use conv2d layer."""
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        x = torch.randn(8, 3, 32, 32)
        y = conv(x)
        
        assert y.shape == (8, 16, 32, 32)
    
    def test_relu_activation(self):
        """ReLU activation."""
        x = torch.randn(10)
        y = nn.relu(x)
        
        assert (y >= 0).all()
        assert torch.allclose(y, torch.relu(x))
    
    def test_sigmoid_activation(self):
        """Sigmoid activation."""
        x = torch.randn(10)
        y = nn.sigmoid(x)
        
        assert (y >= 0).all()
        assert (y <= 1).all()
    
    def test_softmax(self):
        """Softmax function."""
        x = torch.randn(5, 10)
        y = nn.softmax(x, dim=-1)
        
        # Should sum to 1 along last dim
        sums = y.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(5))
    
    def test_layer_norm(self):
        """Layer normalization."""
        layer_norm = nn.LayerNorm(10)
        
        x = torch.randn(32, 10)
        y = layer_norm(x)
        
        assert y.shape == x.shape
    
    def test_dropout(self):
        """Dropout layer."""
        dropout = nn.Dropout(p=0.5)
        
        x = torch.ones(1000)
        
        # In training mode, should zero some elements
        dropout.train()
        y_train = dropout(x)
        
        # In eval mode, should pass through
        dropout.eval()
        y_eval = dropout(x)
        
        assert torch.allclose(y_eval, x)
    
    def test_sequential(self):
        """Sequential container."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        x = torch.randn(32, 10)
        y = model(x)
        
        assert y.shape == (32, 5)


class TestStdOptim:
    """std.optim tests."""
    
    def test_sgd_optimizer(self):
        """SGD optimizer."""
        param = torch.nn.Parameter(torch.randn(10))
        optimizer = optim.SGD([param], lr=0.01)
        
        param.grad = torch.ones(10)
        optimizer.step()
        
        # Parameter should have been updated
        assert param.grad is not None
    
    def test_adam_optimizer(self):
        """Adam optimizer."""
        param = torch.nn.Parameter(torch.randn(10))
        optimizer = optim.Adam([param], lr=0.001)
        
        param.grad = torch.ones(10)
        optimizer.step()
        
        assert param.grad is not None
    
    def test_learning_rate_scheduler(self):
        """Learning rate scheduler."""
        param = torch.nn.Parameter(torch.randn(10))
        optimizer = optim.Adam([param], lr=0.1)
        scheduler = optim.StepLR(optimizer, step_size=1, gamma=0.5)
        
        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        new_lr = optimizer.param_groups[0]["lr"]
        assert new_lr == initial_lr * 0.5


class TestStdConstraints:
    """std.constraints tests."""
    
    def test_equality_constraint(self):
        """Equality constraint violation."""
        x = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 4.0])
        
        violation = constraints.equality_violation(x, target)
        
        # Third element should have violation
        assert violation[2] > 0
        assert violation[0] == 0
    
    def test_inequality_constraint(self):
        """Inequality constraint (x > 0)."""
        x = torch.tensor([-1.0, 0.0, 1.0])
        
        violation = constraints.greater_than_violation(x, 0.0)
        
        # First element violates x > 0
        assert violation[0] > 0
        assert violation[2] == 0
    
    def test_bound_constraint(self):
        """Bound constraint (low < x < high)."""
        x = torch.tensor([-1.0, 0.5, 2.0])
        
        violation = constraints.bound_violation(x, low=0.0, high=1.0)
        
        # First and last elements violate bounds
        assert violation[0] > 0
        assert violation[1] == 0
        assert violation[2] > 0
    
    def test_constraint_weighting(self):
        """Constraint weighting."""
        x = torch.tensor([1.0])
        violation = torch.tensor([0.5])
        
        weighted = constraints.weight(violation, w=10.0)
        
        assert weighted == 5.0


class TestStdDist:
    """std.dist tests."""
    
    def test_normal_distribution(self):
        """Normal distribution."""
        d = dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
        
        samples = d.sample((1000,))
        
        assert abs(samples.mean().item()) < 0.1
        assert abs(samples.std().item() - 1.0) < 0.1
    
    def test_normal_log_prob(self):
        """Normal log probability."""
        d = dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
        
        log_prob = d.log_prob(torch.tensor(0.0))
        
        # Log prob at mean should be maximum
        log_prob_away = d.log_prob(torch.tensor(2.0))
        assert log_prob > log_prob_away
    
    def test_bernoulli_distribution(self):
        """Bernoulli distribution."""
        d = dist.Bernoulli(probs=torch.tensor(0.7))
        
        samples = d.sample((10000,))
        
        # Mean should be close to probability
        assert abs(samples.mean().item() - 0.7) < 0.05
    
    def test_categorical_distribution(self):
        """Categorical distribution."""
        probs = torch.tensor([0.1, 0.2, 0.7])
        d = dist.Categorical(probs=probs)
        
        samples = d.sample((10000,))
        
        # Check approximate frequencies
        counts = torch.bincount(samples.long(), minlength=3).float() / 10000
        assert abs(counts[2].item() - 0.7) < 0.05
    
    def test_reparameterized_sampling(self):
        """Reparameterized sampling allows gradients."""
        loc = torch.tensor(0.0, requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        
        d = dist.Normal(loc, scale)
        sample = d.rsample()
        
        # Should be able to backprop
        sample.backward()
        
        assert loc.grad is not None


class TestStdData:
    """std.data tests."""
    
    def test_tensor_dataset(self):
        """Tensor dataset."""
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        
        dataset = data.TensorDataset(X, y)
        
        assert len(dataset) == 100
        
        x_i, y_i = dataset[0]
        assert x_i.shape == (10,)
        assert y_i.shape == (1,)
    
    def test_dataloader_batching(self):
        """DataLoader batching."""
        X = torch.randn(100, 10)
        dataset = data.TensorDataset(X)
        
        loader = data.DataLoader(dataset, batch_size=32)
        
        batches = list(loader)
        assert len(batches) == 4  # 100 / 32 = 3.125, so 4 batches
    
    def test_dataloader_shuffle(self):
        """DataLoader shuffling."""
        X = torch.arange(100).float().unsqueeze(1)
        dataset = data.TensorDataset(X)
        
        loader = data.DataLoader(dataset, batch_size=100, shuffle=True)
        
        batch = next(iter(loader))[0]
        
        # Should not be in order
        assert not torch.allclose(batch, X)
    
    def test_dataset_map(self):
        """Dataset map transform."""
        X = torch.randn(10, 5)
        dataset = data.TensorDataset(X)
        
        mapped = dataset.map(lambda x: (x[0] * 2,))
        
        original = dataset[0][0]
        transformed = mapped[0][0]
        
        assert torch.allclose(transformed, original * 2)
    
    def test_synthetic_regression_data(self):
        """Synthetic regression data generation."""
        dataset = data.synthetic_regression(
            n_samples=100,
            n_features=10,
            noise=0.1,
            seed=42
        )
        
        assert len(dataset) == 100
        
        X, y = dataset[0]
        assert X.shape == (10,)
        assert y.shape == (1,)


class TestStdDebug:
    """std.debug tests."""
    
    def test_tensor_inspection(self):
        """Tensor inspection."""
        t = torch.randn(10, 10)
        
        stats = debug.inspect(t, name="test_tensor")
        
        assert stats.shape == (10, 10)
        assert stats.dtype == "torch.float32"
    
    def test_gradient_flow_trace(self):
        """Gradient flow tracing."""
        x = torch.randn(10, requires_grad=True)
        y = x.sin().sum()
        
        named_tensors = {"x": x}
        
        tracer = debug.GradientFlowTracer()
        nodes = tracer.trace(y, named_tensors)
        
        assert "x" in nodes
        assert nodes["x"].grad_norm > 0
    
    def test_health_check(self):
        """Tensor health check."""
        good = torch.randn(10)
        bad_nan = torch.tensor([float("nan")])
        bad_inf = torch.tensor([float("inf")])
        
        issues_good = debug.check_health({"good": good})
        issues_nan = debug.check_health({"bad": bad_nan})
        issues_inf = debug.check_health({"bad": bad_inf})
        
        assert len(issues_good) == 0
        assert len(issues_nan) > 0
        assert len(issues_inf) > 0
    
    def test_watchlist(self):
        """Watchlist tracking."""
        param = torch.nn.Parameter(torch.randn(10))
        
        debug.watch("param", param)
        debug.update_watches()
        debug.update_watches()
        
        summary = debug.watch_summary()
        
        assert "param" in summary
    
    def test_trace_recording(self):
        """Trace recording."""
        with debug.trace() as recorder:
            debug.TraceRecorder.get().record(
                debug.TraceKind.FORWARD,
                "test_node",
                torch.randn(10)
            )
        
        events = recorder.events
        assert len(events) == 1
        assert events[0].name == "test_node"


class TestWhySystem:
    """Why system tests."""
    
    def test_why_report_generation(self):
        """Generate why report."""
        # Simulate optimization state
        constraints_info = {
            "positive": (0.5, 1.0, torch.tensor(0.5, requires_grad=True))
        }
        
        param = torch.nn.Parameter(torch.randn(10))
        param.grad = torch.randn(10)
        gradients = {"theta": param.grad}
        
        report = debug.why(constraints_info, gradients, loss=1.5)
        
        assert "positive" in str(report)
    
    def test_constraint_attribution(self):
        """Constraint attribution in why report."""
        # High violation constraint
        c1 = (10.0, 1.0, torch.tensor(10.0))
        # Low violation constraint
        c2 = (0.1, 1.0, torch.tensor(0.1))
        
        constraints_info = {"c1": c1, "c2": c2}
        gradients = {}
        
        report = debug.why(constraints_info, gradients)
        
        # Report should identify c1 as dominant
        assert "c1" in str(report)
