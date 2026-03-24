import torch
import torch.nn.functional as F
import fire

import torch
import torch.nn.functional as F


def _pgd_step(
    x: torch.Tensor,
    AtA: torch.Tensor,
    Atb: torch.Tensor,
    lr_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single projected gradient step.

    Args:
        x:     (..., n)    current iterate
        AtA:   (..., n, n) Gram matrix
        Atb:   (..., n)    moment vector
        lr_vec (..., 1)    per-item step size

    Returns:
        x_new: (..., n) updated iterate
        delta: scalar    max per-item step norm (convergence diagnostic)
    """
    # Gradient of ||Ax - b||^2 w.r.t. x is 2(AtA x - Atb); factor of 2
    # absorbed into lr
    grad = (AtA @ x.unsqueeze(-1)).squeeze(-1) - Atb  # (..., n)

    # Gradient step followed by projection onto the non-negative orthant
    x_new = F.relu(x - lr_vec * grad)  # (..., n)

    # Convergence diagnostic: largest step norm across the batch
    delta = (x_new - x).norm(dim=-1).max()  # scalar

    return x_new, delta


def nnls_batched(
    A: torch.Tensor,
    b: torch.Tensor,
    max_iter: int = 1000,
    tol: float = 1e-6,
    lr: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Batched differentiable Non-Negative Least Squares.

    Solves min_{x >= 0} ||Ax - b||^2 for each item in the batch via
    projected gradient descent. The iterative solve runs under no_grad;
    a single differentiable step at the converged point provides
    gradients w.r.t. A and b.

    Args:
        A:        (..., m, n) batched matrix
        b:        (..., m)    batched vector
        max_iter: maximum number of projected gradient iterations
        tol:      convergence tolerance on max per-item step norm
        lr:       step size; (...,) tensor or scalar. Defaults to
                  1 / λ_max(AᵀA) per item, which guarantees descent.

    Returns:
        x: (..., n) non-negative solution
    """

    # Precompute the Gram matrix and moment vector. Both are (..., n, n)
    # and (..., n) respectively, preserving all batch dimensions.
    AtA = A.transpose(-1, -2) @ A  # (..., n, n)
    Atb = (A.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)  # (..., n)

    # Per-item step size. 1 / λ_max(AᵀA) is the largest step that
    # guarantees the gradient step decreases the objective.
    if lr is None:
        L = torch.linalg.eigvalsh(AtA).amax(dim=-1).clamp(min=1e-12)
        lr_vec = (1.0 / L).unsqueeze(-1)  # (..., 1)
    elif isinstance(lr, torch.Tensor):
        lr_vec = lr.double().unsqueeze(-1)  # (..., 1)
    else:
        lr_vec = torch.full(
            (*A.shape[:-2], 1), lr, dtype=torch.float64, device=A.device
        )  # (..., 1)

    # Initialise at zero, which is feasible (satisfies x >= 0)
    x = torch.zeros_like(Atb)  # (..., n)

    # Iterative solve. no_grad keeps the loop out of the autograd graph;
    # gradients are provided by the differentiable step below instead.
    with torch.no_grad():
        for _ in range(max_iter):
            x, delta = _pgd_step(x, AtA, Atb, lr_vec)
            # delta is the max step norm across the batch; we stop only
            # when every item has converged
            if delta.item() < tol:
                break

    # One differentiable step at the converged x. Because x is treated as
    # a constant here (it was produced under no_grad), the gradient flows
    # directly through A and b without backpropagating through the loop.
    residual = (A @ x.unsqueeze(-1)).squeeze(-1) - b  # (..., m)
    grad = (A.transpose(-1, -2) @ residual.unsqueeze(-1)).squeeze(-1)  # (..., n)
    x = F.relu(x - lr_vec * grad)  # (..., n)

    return x


def main():

    torch.manual_seed(0)

    batch, m, n = 4, 20, 5
    x_true = torch.rand(batch, n) * 3

    A = torch.randn(batch, m, n, requires_grad=True)
    b = (A.detach() @ x_true.unsqueeze(-1)).squeeze(-1) + 0.01 * torch.randn(batch, m)

    x_eager = nnls_batched(A, b)

    print(f"Error in eager solution: {torch.linalg.norm(x_true - x_eager)}")

    # # Compiled (first call traces, subsequent calls run the cached graph)
    # x_compiled = nnls_batched_compiled(A, b)

    # print("Max deviation eager vs compiled:", (x_eager - x_compiled).abs().max().item())

    # Gradients still work
    loss = (A @ x_eager.float().unsqueeze(-1)).squeeze(-1).sub(b).pow(2).sum()
    loss.backward()
    print("dL/dA shape:", A.grad.shape)  # (8, 20, 5)


if __name__ == "__main__":
    fire.Fire(main)
