from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    f_plus_eps = f(*(vals[:arg] + (vals[arg] + epsilon,) + vals[arg + 1 :]))
    f_minus_eps = f(*(vals[:arg] + (vals[arg] - epsilon,) + vals[arg + 1 :]))
    return (f_plus_eps - f_minus_eps) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the computational graph with respect to this variable.

        Args:
        ----
            x (Any): The derivative to be accumulated.

        Returns:
        -------
            None

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable.

        Returns
        -------
            int: A unique integer identifier.

        """
        ...

    def is_leaf(self) -> bool:
        """Checks if this variable is a leaf node in the computational graph.

        Returns
        -------
            bool: True if the variable is a leaf node, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Checks if this variable is a constant.

        Returns
        -------
            bool: True if the variable is a constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables in the computational graph.

        Returns
        -------
            Iterable["Variable"]: An iterable of parent variables.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for this variable's parents.

        Args:
        ----
            d_output (Any): The gradient of the output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, each containing a parent variable and its gradient.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    l = []
    t = []
    path = []

    def visit(v: Variable) -> None:
        if v.unique_id in l:
            return
        if v.unique_id in t:
            raise ValueError("Cyclic graph detected")
        t.append(v.unique_id)
        if not v.is_constant():
            for parent in v.parents:
                visit(parent)
        l.insert(0, v.unique_id)  # all children (parents in reverse) visited
        path.insert(0, v)

    visit(variable)

    return path


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: The derivative of the final output with respect to this variable

    Returns:
    -------
    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    deriv_map = {variable.unique_id: deriv}
    for v in topological_sort(variable):
        if v.is_constant():
            # skip constant variables
            continue
        if v.is_leaf():
            v.accumulate_derivative(deriv_map[v.unique_id])
        else:
            for parent, deriv in v.chain_rule(deriv_map[v.unique_id]):
                if parent.unique_id not in deriv_map:
                    deriv_map[parent.unique_id] = 0
                deriv_map[parent.unique_id] += deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved values."""
        return self.saved_values
