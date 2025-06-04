# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch


def min_power_normalization(logits: torch.Tensor, power: float = 2.0, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Normalizes logits using a minimum power normalization approach.

    This function applies power normalization to the input logits, raising them to a power
    and normalizing to create a probability distribution. If the sum is too small (below epsilon),
    returns zeros to avoid division by very small numbers.

    Args:
        logits: Input tensor to be normalized
        power: Power to raise the logits to. Defaults to 2.0.
        epsilon: Small value to prevent division by zero. Defaults to 1e-8.

    Returns:
        Normalized probabilities
    """
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)

    powered_logits = logits**power
    sum_powered = torch.sum(powered_logits)
    if sum_powered > epsilon:
        probabilities = powered_logits / sum_powered
    else:
        probabilities = torch.zeros_like(powered_logits)

    return probabilities


def sign_preserving_multiplication(a: float, b: float) -> float:
    """
    Multiplies two numbers while preserving sign properties.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Result with sign preservation logic
    """
    return -abs(a) * abs(b) if a < 0 or b < 0 else a * b 