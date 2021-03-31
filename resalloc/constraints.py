import abc

import torch


class Constraint(abc.ABC):
    """A generic constraint.

    To create a custom constraint, create a subclass of this class,
    and implement its abstract methods.
    """

    @abc.abstractmethod
    def name(self) -> str:
        """The name of the constraint."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialization(self, device=None) -> torch.Tensor:
        """Return a random point in the constraint set.

        Arguments
        ---------
        device: str
            Device on which to store the returned embedding.

        Returns
        -------
        torch.Tensor
            a tensor of shape satisfying the constraints.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def project(self, Z: torch.Tensor, inplace=False) -> torch.Tensor:
        """Project ``Z`` onto the constraint set.

        Returns a projection of ``Z`` onto the constraint set.


        Arguments
        ---------
        Z: torch.Tensor
            The point to project.
        inplace: bool
            If True, stores the projection in ``Z``.

        Returns
        -------
        The projection of ``Z`` onto the constraints.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def project_onto_tangent_cone(
        self, X: torch.Tensor, Z: torch.Tensor, inplace=False
    ) -> torch.Tensor:
        """Project ``Z`` onto the tangent space of the constraint set at ``X``.

        Returns the Euclidean projection of ``Z`` onto
        the tangent space of the constraint set at ``X`` (where ``X`` is
        some matrix satisfying the constraints).

        ``X`` and ``Z`` should have the same shape.

        Arguments
        ---------
        X: torch.Tensor
            A point satisfying the constraints.
        Z: torch.Tensor
            The point to project.
        inplace: bool
            If True, stores the projection in ``Z``.

        Return
        ------
        The projection of ``Z`` onto the tangent space of the constraint
        set at ``X``.
        """
        raise NotImplementedError


class Nonnegative(Constraint):
    def __init__(self, shape):
        super(Nonnegative, self).__init__()
        self.shape = shape

    def name(self):
        return "nonnegative"

    def initialization(self, device=None):
        return torch.randn(self.shape, device=device).abs()

    def project(self, Z, inplace=False):
        # nonpositive entries are made 0
        mask = Z <= 0.0
        if not inplace:
            Z = Z.clone()
        Z[mask] = 0.0
        return Z

    def project_onto_tangent_cone(self, X, Z, inplace=False):
        if (X > 0.0).all():
            # in the interior of the nonnegative orthant, any direction is fine
            return Z

        # X is on the boundary, with at least one component being nonpositive
        #
        # find the coordinates on the boundary, ie with X <= 0 ...
        # for these coordinates, only positive directions are allowed
        mask = X <= 0
        if not inplace:
            Z = Z.clone()
        Z[mask] = torch.max(Z[mask], torch.tensor(0.0, device=Z.device))
        return Z
