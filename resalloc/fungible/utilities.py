import torch


def _initial_prices(utility, alloc_problem):
    x = alloc_problem.resource_limits[1:] / alloc_problem.n_jobs
    if x.sum() > 1:
        x = x / x.sum()
    t = alloc_problem.A[:, 1:] @ x
    u_prime_t = utility._derivative(t)
    prices = (u_prime_t[:, None] * alloc_problem.A[:, 1:]).sum(
        axis=0
    ) / alloc_problem.n_jobs
    prices = torch.hstack(
        [
            torch.tensor(
                [0.0],
                device=alloc_problem.A.device,
                dtype=alloc_problem.A.dtype,
            ),
            prices,
        ]
    )
    return prices
 
# TODO: torch.nn.modules
# TODO: exponential utility
class Log(object):
    def __call__(self, t):
        return torch.log(t)

    def _derivative(self, t):
        return 1 / t

    def initial_prices(self, alloc_problem):
        return _initial_prices(self, alloc_problem)

    def argmax(self, slopes, alloc_problem):
        del alloc_problem
        return 1.0 / slopes

    def cvxpy_utility(self, t):
        import cvxpy as cp
        return cp.log(t)


class Power(object):
    def __init__(self, exponent):
        if exponent >= 1 or exponent == 0:
            raise ValueError(
                "The exponent should be in (0, 1) or less than 0, "
                f"but received exponent={exponent}"
            )
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent).float()
        self.exponent = exponent
        self.sign = -1 if exponent < 0 else 1

        self.derivative_exponent = self.exponent - 1
        self.derivative_coefficient = self.sign * self.exponent
        self.inverse_exponent = 1 / derivative_exponent
        self.inverse_coefficient = self.derivative_coefficient.pow(
            inverse_exponent
        )

    def _derivative(self, t):
        return self.derivative_coefficient * t.pow(self.derivative_exponent)

    def initial_prices(self, alloc_problem):
        return _initial_prices(self, alloc_problem)

    def __call__(self, t):
        return self.sign * t.pow(self.exponent)

    def argmax(self, slopes, alloc_problem):
        return self.inverse_coefficient * slopes.pow(self.inverse_exponent)

    def cvxpy_utility(self, t):
        pass


class Linear(object):
    def __call__(self, t):
        return t

    def initial_prices(self, alloc_problem):
        del alloc_problem
        prices = torch.ones(alloc_problem.resource_limits.shape)
        prices[0] = 0
        return prices

    def argmax(self, slopes, alloc_problem):
        sign_mask = (1 - slopes) < 0
        # or min and mask?
        ts = torch.zeros(
            alloc_problem.a_lhs.shape, device=slopes.device, dtype=slopes.dtype
        )
        ts[sign_mask] = torch.min(
            alloc_problem.a_lhs[sign_mask], alloc_problem.a_rhs[sign_mask]
        )
        ts[~sign_mask] = torch.max(
            alloc_problem.a_lhs[~sign_mask], alloc_problem.a_rhs[~sign_mask]
        )
        return ts

    def cvxpy_utility(self, t):
        return t


class TargetPriority(object):
    def __init__(self, targets, priorities):
        self.targets = targets
        self.priorities = priorities

    def _derivative(self, t):
        # slopes = self.priorities.clone()
        # slopes[mask] = 0.0
        return self.priorities.squeeze() * t

    def __call__(self, t):
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        return self.priorities * torch.minimum(
            t - self.targets, torch.tensor(0.0, device=t.device, dtype=t.dtype)
        )

    def initial_prices(self, alloc_problem):
        return _initial_prices(self, alloc_problem)

    def argmax(self, slopes, alloc_problem):
        gamma = alloc_problem.a_min
        delta = alloc_problem.a_max
        branch_one_mask = (
            (self.priorities >= slopes)
            * (self.targets >= gamma)
            * (self.targets <= delta)
        )

        branch_two_mask = self.priorities <= slopes

        branch_three_mask = (self.priorities > slopes) * (self.targets >= delta)

        # TODO: check branches are exhaustive

        t = torch.zeros(slopes.shape, dtype=slopes.dtype, device=slopes.device)
        t[branch_one_mask] = self.targets.repeat((1, t.shape[1]))[
            branch_one_mask
        ]
        t[branch_two_mask] = gamma[branch_two_mask]
        t[branch_three_mask] = delta[branch_three_mask]

        return t

    def cvxpy_utility(self, t):
        import cvxpy as cp
        return cp.multiply(
            self.priorities.cpu().numpy().squeeze(),
            cp.minimum(t - self.targets.cpu().numpy().squeeze(), 0.0),
        )
