import logging
import sys
import time

import torch

from resalloc.lbfgs import LBFGS


LOGGER = logging.getLogger("__doubly_projected__")
LOGGER.propagate = False
LOGGER.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
_formatter = logging.Formatter(
    fmt="%(asctime)s: %(message)s", datefmt="%b %d %I:%M:%S %p"
)
_stream_handler.setFormatter(_formatter)
LOGGER.addHandler(_stream_handler)


class SolveStats(object):
    """Summary statistics for a solve.

    Attributes
    ----------
    values: sequence
        The objective function values at each iteration.
    residual_norms: sequence
        The residual norm at each iteration.
    step_size_percents: sequence
        The relative size of each step.
    solve_time: float
        The time it took to create the embedding, in seconds.
    snapshots: sequence
        Snapshots of the embedding.
    snapshot_every: int
        The number of iterations between snapshots.
    """

    def __init__(
        self,
        values,
        residual_norms,
        step_size_percents,
        solve_time,
        times,
        snapshots,
        snapshot_every,
    ):
        self.values = values
        self.residual_norms = residual_norms
        self.step_size_percents = step_size_percents
        self.solve_time = solve_time
        self.iterations = len(values)
        self.times = times
        self.snapshots = snapshots
        self.snapshot_every = snapshot_every

    def __str__(self):
        return (
            "SolveStats:\n"
            "\tvalue {0:.3g}\n"
            "\tresidual norm {1:.3g}\n"
            "\tsolve_time (s) {2:.3g}\n"
            "\titerations {3}".format(
                self.values[-1],
                self.residual_norms[-1],
                self.solve_time,
                self.iterations,
            )
        )

    def _repr_pretty_(self, p, cycle):
        del cycle
        text = self.__str__()
        p.text(text)


def lbfgs(
    X,
    objective_fn,
    constraint,
    eps=1e-5,
    max_iter=300,
    memory_size=10,
    use_line_search=True,
    project_gradients=True,
    use_cached_loss=True,
    verbose=False,
    print_every=None,
    snapshot_every=None,
    outer_callback=None,
    verbose_callback=None,
    stopping_criterion=None,
):
    start_time = time.time()
    values = []
    grad_norms = []
    step_size_percents = []
    times = []
    snapshots = []

    if print_every is None:
        print_every = max(1, max_iter // 10)

    # LBFGS set-up
    #
    # This callback logs the loss value and gradient norm per teration
    def callback(loss, grad):
        values.append(loss.detach().cpu().item())
        if not project_gradients:
            with torch.no_grad():
                grad = -constraint.project_onto_tangent_cone(
                    X, -grad, inplace=False
                )
        grad_norms.append(grad.detach().norm(p="fro").cpu().item())

    # closure for LBFGS optimization step; projects gradient onto
    # onto the tangent cone of the constraintsat the iterate
    def value_and_grad():
        opt.zero_grad()
        loss = objective_fn(X)
        loss.backward()
        if project_gradients:
            with torch.no_grad():
                X.grad.mul_(-1.)
                minus_grad = constraint.project_onto_tangent_cone(
                    X, X.grad, inplace=True
                )
                X.grad.mul_(-1.)
        return loss

    ls = "strong_wolfe" if use_line_search else None
    opt = LBFGS(
        params=[X],
        max_iter=1,
        history_size=memory_size,
        line_search_fn=ls,
        callback=callback,
        tolerance_grad=-torch.tensor(
            float("inf"), dtype=X.dtype, device=X.device
        ),
        tolerance_change=-torch.tensor(
            float("inf"), dtype=X.dtype, device=X.device
        ),
        project_callback=lambda X: constraint.project(X, inplace=True),
        use_cached_loss=use_cached_loss,
    )

    digits = len(str(max_iter))
    start = time.time()
    for iteration in range(0, max_iter):
        if snapshot_every is not None and iteration % snapshot_every == 0:
            snapshots.append(X.detach().cpu().clone())
        with torch.no_grad():
            norm_X = X.norm(p="fro")
        X.requires_grad_(True)
        opt.step(value_and_grad)
        X.requires_grad_(False)

        with torch.no_grad():
            constraint.project(X, inplace=True)

        times.append(time.time() - start)
        try:
            h = opt.state[opt._params[0]]["t"]
            d = opt.state[opt._params[0]]["d"]
            percent_change = 100 * h * d.norm() / norm_X
        except KeyError:
            h = 0.0
            d = 0.0
            percent_change = 0.0
        step_size_percents.append(float(percent_change))

        norm_grad = grad_norms[-1]
        if outer_callback is not None:
            outer_callback(values[-1])

        if verbose and (
            ((iteration % print_every == 0)) or (iteration == max_iter - 1)
        ):
            if verbose_callback is not None:
                verbose_callback(LOGGER, "iteration %0*d | " % (digits, iteration))
            else:
                LOGGER.info(
                    "iteration %0*d | value %6f | residual norm %g | "
                    "step length %g | percent change %g"
                    % (
                        digits,
                        iteration,
                        values[-1],
                        norm_grad,
                        h,
                        percent_change,
                    )
                )
        if stopping_criterion is not None:
            residual = stopping_criterion()
        else:
            residual = norm_grad

        if residual <= eps:
            if verbose:
                LOGGER.info(
                    "Converged in %03d iterations, with residual %g"
                    % (iteration + 1, residual)
                )
            break
        elif h == 0:
            opt.reset()
    tot_time = time.time() - start_time
    solve_stats = SolveStats(
        values,
        grad_norms,
        step_size_percents,
        tot_time,
        times,
        snapshots,
        snapshot_every,
    )
    return X, solve_stats


def gradient_descent(
    X,
    objective_fn,
    constraint,
    step_length=1e-4,
    eps=1e-5,
    max_iter=300,
    project_gradients=True,
    verbose=False,
    print_every=None,
    snapshot_every=None,
):
    start_time = time.time()
    values = []
    grad_norms = []
    step_size_percents = []
    times = []
    snapshots = []

    if print_every is None:
        print_every = max(1, max_iter // 10)

    # This callback logs the loss value and gradient norm per teration
    def callback(loss, grad):
        values.append(loss.detach().cpu().item())
        if not project_gradients:
            with torch.no_grad():
                grad = -constraint.project_onto_tangent_cone(
                    X, -grad, inplace=False
                )
        grad_norms.append(grad.detach().norm(p="fro").cpu().item())

    # closure for optimization step; projects gradient onto
    # onto the tangent cone of the constraint at the iterate
    def value_and_grad():
        X.grad = None
        loss = objective_fn(X)
        loss.backward()
        if project_gradients:
            with torch.no_grad():
                X.grad.mul_(-1.)
                minus_grad = constraint.project_onto_tangent_cone(
                    X, X.grad, inplace=True
                )
                X.grad.mul_(-1.)
        return loss

    digits = len(str(max_iter))
    start = time.time()
    for iteration in range(0, max_iter):
        if snapshot_every is not None and iteration % snapshot_every == 0:
            snapshots.append(X.detach().cpu().clone())
        with torch.no_grad():
            norm_X = X.norm(p="fro")
        X.requires_grad_(True)
        loss = value_and_grad()
        callback(loss, X.grad)
        X.requires_grad_(False)

        with torch.no_grad():
            X = X.sub_(step_length*X.grad)
            constraint.project(X, inplace=True)

        times.append(time.time() - start)
        d = X.grad
        percent_change = 100 * step_length * d.norm() / norm_X
        step_size_percents.append(float(percent_change))

        norm_grad = grad_norms[-1]
        if verbose and (
            ((iteration % print_every == 0)) or (iteration == max_iter - 1)
        ):
            LOGGER.info(
                "iteration %0*d | value %6f | residual norm %g | "
                "step length %g | percent change %g"
                % (
                    digits,
                    iteration,
                    values[-1],
                    norm_grad,
                    step_length,
                    percent_change,
                )
            )
        if norm_grad <= eps:
            if verbose:
                LOGGER.info(
                    "Converged in %03d iterations, with residual norm %g"
                    % (iteration + 1, norm_grad)
                )
            break
    tot_time = time.time() - start_time
    solve_stats = SolveStats(
        values,
        grad_norms,
        step_size_percents,
        tot_time,
        times,
        snapshots,
        snapshot_every,
    )
    return X, solve_stats
