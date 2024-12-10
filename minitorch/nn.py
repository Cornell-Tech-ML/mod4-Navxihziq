from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    output = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    output = output.permute(0, 1, 2, 4, 3, 5)
    output = output.contiguous().view(batch, channel, new_height, new_width, kh * kw)
    return output, new_height, new_width


# TODO: Implement for Task 4.4[?].
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling 2D"""
    output, new_height, new_width = tile(input, kernel)
    return output.mean(dim=4).view(
        input.shape[0], input.shape[1], new_height, new_width
    )


max_reduce = FastOps.reduce(operators.max, float("-inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor"""
    return input == max_reduce(input, dim)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:  # noqa: D102
        max_val = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, max_val)
        return max_val

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:  # noqa: D102
        input, max_val = ctx.saved_tensors
        grad_input = (input == max_val) * grad_output
        return grad_input, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction"""
    return Max.apply(input, input._ensure_tensor(dim))


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling 2D"""
    output, new_height, new_width = tile(input, kernel)
    return max(output, 4).view(input.shape[0], input.shape[1], new_height, new_width)


def softmax(input: Tensor, dim: int) -> Tensor:
    """Apply softmax"""
    return input.exp() / input.exp().sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Apply log softmax
    $log(softmax(x)) = x - m - log(sum(exp(x-m)))$
    """
    m = max(input, dim)
    return input - m - (input - m).exp().sum(dim=dim).log()


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout"""
    if ignore:
        return input
    mask = rand(input.shape) > rate
    return input * mask
