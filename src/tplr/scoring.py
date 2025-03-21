import torch


def min_power_normalization(
    logits: torch.Tensor, power: float = 2.0, epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Normalizes logits using a minimum power normalization approach.
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


def update_final_score(
    prev_final_score: float,
    gradient_score: float,
    binary_moving_average: float,
    final_score_ma_alpha: float,
) -> float:
    """
    Calculates the new final moving average score.

    The binary moving average is normalized by dividing by 2, then used with the gradient score,
    and then incorporated into the moving average update.
    """
    norm_binary = binary_moving_average / 2.0
    final_score = gradient_score * norm_binary
    return max(
        final_score_ma_alpha * prev_final_score
        + (1 - final_score_ma_alpha) * final_score,
        0.0,
    )


def compute_improvement_metrics(
    loss_before_own: float,
    loss_after_own: float,
    loss_before_random: float,
    loss_after_random: float,
):
    """
    Computes loss improvements and returns:
      (relative_improvement_own, relative_improvement_random, gradient_score, binary_indicator)
    """
    loss_improvement_own = loss_before_own - loss_after_own
    relative_improvement_own = (
        (loss_improvement_own / loss_before_own) if loss_before_own > 0 else 0.0
    )

    loss_improvement_random = loss_before_random - loss_after_random
    relative_improvement_random = (
        (loss_improvement_random / loss_before_random)
        if loss_before_random > 0
        else 0.0
    )

    # Here we treat the gradient_score simply as the relative improvement on own data.
    gradient_score = relative_improvement_own
    binary_indicator = (
        1 if relative_improvement_own > relative_improvement_random else -1
    )
    return (
        relative_improvement_own,
        relative_improvement_random,
        gradient_score,
        binary_indicator,
    )
