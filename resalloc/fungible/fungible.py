import time
import numpy as np
import torch

import resalloc.constraints as constraints
import resalloc.optim as optim
import resalloc.fungible.utilities as utilities


class _Allocator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prices, allocator_object):
        value, gradient = allocator_object._evaluate_dual_function(prices)
        ctx.save_for_backward(gradient)
        return value

    @staticmethod
    def backward(ctx, dprices):
        (gradient,) = ctx.saved_tensors
        return gradient, None


_make_allocation = _Allocator.apply


# TODO torch.nn.Module
class AllocationProblem(object):
    def __init__(
        self,
        throughput_matrix,
        resource_limits,
        utility_function,
        job_demands=None,
    ):
        self.utility_fn = utility_function

        self.A = torch.hstack(
            [
                torch.zeros(
                    (throughput_matrix.shape[0], 1),
                    device=throughput_matrix.device,
                ),
                throughput_matrix,
            ]
        )
        self.n_jobs = throughput_matrix.shape[0]
        self.n_resources = throughput_matrix.shape[1]

        self.set_resource_limits(resource_limits)

        self.resource_pairs = torch.triu_indices(
            self.n_resources + 1, self.n_resources + 1, 1, device=self.A.device
        ).T

        self.lhs = self.resource_pairs[:, 0]
        self.rhs = self.resource_pairs[:, 1]

        # shape (n_jobs, n_resources + 1 choose 2)
        self.a_lhs = self.A[:, self.lhs]
        self.a_rhs = self.A[:, self.rhs]

        self.a_denom = self.a_rhs - self.a_lhs
        self.a_denom_zero_mask = self.a_denom == 0

        if (self.a_denom == 0).any():
            print(
                "Warning: some rows of the throughput matrix "
                "have duplicated entries."
            )

        self.a_min = torch.min(self.a_lhs, self.a_rhs)
        self.a_max = torch.max(self.a_lhs, self.a_rhs)

        self.job_indices = torch.arange(self.A.shape[0])

        if job_demands is not None and job_demands.ndim == 1:
            job_demands = job_demands.reshape(job_demands.numel(), -1)
        if job_demands is not None and job_demands.shape[1] > 1:
            # add an entry for the slack ...
            job_demands = torch.hstack([torch.zeros((self.n_jobs, 1)), job_demands])
        self.job_demands = job_demands

    def set_resource_limits(self, resource_limits):
        # first entry is a slack
        resource_limits = torch.hstack(
            [
                torch.tensor([np.inf], device=resource_limits.device),
                resource_limits,
            ]
        )
        self.resource_limits = resource_limits

    def resources_used(self, X):
        if self.job_demands is None:
            return X.sum(axis=0)
        else:
            return (X * self.job_demands).sum(axis=0)

    def _evaluate_dual_function(self, prices):
        p_lhs = prices[self.lhs]
        p_rhs = prices[self.rhs]

        if self.job_demands is not None:
            if self.job_demands.shape[1] == 1:
                p_lhs = (
                    p_lhs[None, :].repeat(self.A.shape[0], 1) * self.job_demands
                )
                p_rhs = (
                    p_rhs[None, :].repeat(self.A.shape[0], 1) * self.job_demands
                )
            else:
                p_lhs = (
                    p_lhs[None, :].repeat(self.A.shape[0], 1)
                    * self.job_demands[:, self.lhs]
                )
                p_rhs = (
                    p_rhs[None, :].repeat(self.A.shape[0], 1)
                    * self.job_demands[:, self.rhs]
                )

        slopes = (p_rhs - p_lhs) / self.a_denom
        if self.a_denom_zero_mask.any():
            # if two entries of a_i are the same, then only the cheaper
            # of the two resources will be used, and c(t) = price * t / a
            # (same as the case when the slack is used)
            cheapest_prices = torch.minimum(p_lhs, p_rhs).expand(
                self.n_jobs, -1
            )
            slopes[self.a_denom_zero_mask] = (
                cheapest_prices[self.a_denom_zero_mask]
                * self.a_lhs[self.a_denom_zero_mask]
            )

        ts = self.utility_fn.argmax(slopes, self)
        ts = torch.max(torch.min(ts, self.a_max), self.a_min)

        x_i = (self.a_rhs - ts) / self.a_denom
        x_j = (ts - self.a_lhs) / self.a_denom

        if self.a_denom_zero_mask.any():
            # TODO there's surely a more efficient way to batch this
            lhs_cheaper_mask = p_lhs.expand(self.n_jobs, -1) < p_rhs.expand(
                self.n_jobs, -1
            )
            if (lhs_cheaper_mask).any():
                mask = self.a_denom_zero_mask * lhs_cheaper_mask
                x_i[mask] = ts[mask] / self.a_lhs[mask]
                x_j[mask] = 0
            if (~lhs_cheaper_mask).any():
                mask = self.a_denom_zero_mask * ~lhs_cheaper_mask
                x_i[mask] = 0
                x_j[mask] = ts[mask] / self.a_lhs[mask]

        assert (x_i >= 0).all()
        assert (x_j >= 0).all()

        costs = p_lhs * x_i + p_rhs * x_j
        utilities = self.utility_fn(ts)
        excess = utilities - costs
        excess[torch.isnan(excess)] = -float("inf")

        # the optimal excess utility, ie the optimal value of the subproblems
        dual_function_values = torch.max(excess, axis=1).values

        X = torch.zeros(self.A.shape, device=self.A.device, dtype=self.A.dtype)
        opt_indices = torch.argmax(excess, axis=1)
        sparsity_pattern = self.resource_pairs[opt_indices]
        X[self.job_indices, sparsity_pattern[:, 0]] = x_i[
            self.job_indices, opt_indices
        ]
        X[self.job_indices, sparsity_pattern[:, 1]] = x_j[
            self.job_indices, opt_indices
        ]
        self.X = X

        resources_used = self.resources_used(X)
        gradient = self.resource_limits - resources_used
        gradient[0] = 0

        dual_function_value = (
            dual_function_values.sum().item()
            + (prices[1:] * self.resource_limits[1:]).sum()
        )
        return dual_function_value, gradient

    def evaluate_dual_function(self, prices):
        return _make_allocation(prices, self)

    def make_feasible(self, allocation):
        """Returns a feasible allocation (wrt resource limit)"""
        allocation = allocation.clone()
        resource_usage = self.resources_used(allocation)
        violated_resources = resource_usage > self.resource_limits
        if violated_resources.any():
            allocation[:, violated_resources] *= (
                self.resource_limits[violated_resources]
                / resource_usage[violated_resources]
            )
        return allocation

    def utility(self, allocation):
        utility = (
            self.utility_fn((self.A * allocation).sum(axis=1))
            .sum()
            .cpu()
            .item()
        )
        return utility

    def make_cvxpy_problem(self):
        import cvxpy as cp

        X = cp.Variable(self.A.shape, nonneg=True)
        throughput = cp.sum(cp.multiply(self.A.cpu().numpy(), X), axis=1)
        # maybe divide by n_jobs, may yield better scaling / condition number ...
        utility = cp.sum(self.utility_fn.cvxpy_utility(throughput))

        if self.job_demands is not None:
            if self.job_demands.shape[1] == 1:
                job_demands = (
                    self.job_demands.repeat(1, X.shape[1] - 1).cpu().numpy()
                )
            else:
                job_demands = self.job_demands.cpu().numpy()[:, 1:]

            resource_used = cp.sum(cp.multiply(job_demands, X[:, 1:]), axis=0)
        else:
            resource_used = cp.sum(X[:, 1:], axis=0)

        problem = cp.Problem(
            cp.Maximize(utility),
            [
                cp.sum(X, axis=1) <= 1,
                resource_used <= self.resource_limits[1:].cpu().numpy(),
            ],
        )
        return problem

    def solve(
        self,
        eps=1e-3,
        max_iter=50,
        prices=None,
        snapshot_allocations=False,
        snapshot_throughputs=False,
        snapshot_resource_usage=False,
        print_every=None,
        verbose=False,
    ):
        """Solves the allocation problem

        Returns the vector of optimal prices, and an object holding
        various statistics about the solve.

        The optimal allocation is saved to the ``X`` attribute of this
        object.

        TODO document ...
        """
        return _solve(
            self,
            eps=eps,
            max_iter=max_iter,
            prices=prices,
            snapshot_allocations=snapshot_allocations,
            snapshot_throughputs=snapshot_throughputs,
            snapshot_resource_usage=snapshot_resource_usage,
            print_every=print_every,
            verbose=verbose,
        )


def _solve(
    problem,
    eps=1e-3,
    max_iter=50,
    prices=None,
    snapshot_allocations=False,
    snapshot_throughputs=False,
    snapshot_resource_usage=False,
    print_every=None,
    verbose=False,
):
    constraint = constraints.Nonnegative((problem.n_resources,))

    def objective_fn(prices):
        return problem.evaluate_dual_function(prices)

    if prices is None:
        prices = problem.utility_fn.initial_prices(problem)
    else:
        prices = torch.hstack(
            [
                torch.tensor([0.0], device=prices.device, dtype=prices.dtype),
                prices,
            ]
        )

    dual_values = []
    utils = []
    gaps = []
    rel_gaps = []

    allocations = []
    throughputs = []
    resource_usage = []

    def callback(dual_value):
        X_feas = problem.make_feasible(problem.X)
        dual_values.append(dual_value / problem.n_jobs)
        utils.append(problem.utility(X_feas) / problem.n_jobs)
        gaps.append(dual_values[-1] - utils[-1])
        rel_gaps.append(gaps[-1] / np.abs(dual_values[-1]))

        if snapshot_allocations:
            allocations.append(problem.X.cpu().clone())
        if snapshot_throughputs:
            throughputs.append((problem.A * X_feas).sum(axis=1))
        if snapshot_resource_usage:
            resource_usage.append(problem.resources_used(problem.X))

    def verbose_callback(logger, header):
        logger.info(
            f"{header}utility={utils[-1]:g} | dual_value={dual_values[-1]:g} | "
            f"gap={gaps[-1]:.2e}"
        )

    def stopping_criterion():
        return gaps[-1]

    prices, stats = optim.lbfgs(
        prices,
        objective_fn,
        constraint,
        eps=eps,
        max_iter=max_iter,
        project_gradients=True,
        verbose=verbose,
        print_every=print_every,
        snapshot_every=1,
        outer_callback=callback,
        verbose_callback=verbose_callback,
        stopping_criterion=stopping_criterion,
    )

    stats.dual_values = dual_values
    stats.utils = utils
    stats.gaps = gaps
    stats.rel_gaps = rel_gaps

    stats.allocations = allocations
    stats.throughputs = throughputs
    stats.resource_usage = resource_usage

    return prices, stats
