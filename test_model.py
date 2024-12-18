import torch
import pytest
from model import CIFAR10Net, DepthwiseSeparableConv

def test_model_output_shape():
    model = CIFAR10Net()
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    output = model(x)
    assert output.shape == (batch_size, 10), f"Expected shape {(batch_size, 10)}, got {output.shape}"

def test_depthwise_separable_conv():
    conv = DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=3)
    x = torch.randn(1, 64, 32, 32)
    output = conv(x)
    assert output.shape == (1, 128, 32, 32), f"Expected shape (1, 128, 32, 32), got {output.shape}"

def test_model_parameter_count():
    model = CIFAR10Net()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 200000, f"Model has {param_count} parameters, should be less than 200000"

def test_forward_pass():
    model = CIFAR10Net()
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(x)
    assert not torch.isnan(output).any(), "Model output contains NaN values"
    assert not torch.isinf(output).any(), "Model output contains infinite values"

def test_model_devices():
    model = CIFAR10Net()
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(1, 3, 32, 32).cuda()
    else:
        x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.device == x.device, "Model output device doesn't match input device" 