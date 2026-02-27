"""Autograd stuff."""
import math
import random


class Value:
    """Let there be Autograd to recursively apply the chain rule through a computation graph."""

    __slots__ = ('data', 'grad', '_children', '_local_grads')  # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        """Make new instance."""
        self.data = data  # scalar value of this node calculated during forward pass
        self.grad = 0  # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children  # children of this node in the computation graph
        self._local_grads = local_grads  # local derivative of this node w.r.t. its children

    def __add__(self, other):
        """Define + operator."""
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        """Define * operator."""
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        """Define ** operator."""
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        """Define math.log method."""
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        """Define math.exp method."""
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        """Define relu method."""
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        """Define - operator."""
        return self * -1

    def __radd__(self, other):
        """Define right + operator."""
        return self + other

    def __sub__(self, other):
        """Define - operator."""
        return self + (-other)

    def __rsub__(self, other):
        """Define right - operator."""
        return other + (-self)

    def __rmul__(self, other):
        """Define right * operator."""
        return self * other

    def __truediv__(self, other):
        """Define truediv operator."""
        return self * other**-1

    def __rtruediv__(self, other):
        """Define truediv operator."""
        return other * self**-1

    def backward(self):
        """Make back propogation."""
        topo = []
        visited = set()

        def build_topo(v):
            """Make topo."""
            if v not in visited:
                visited.add(v)
                for child in v._children:  # pylint: disable=protected-access
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):  # pylint: disable=protected-access
                child.grad += local_grad * v.grad


def matrix(nout, nin, std=0.08):
    """Return matrix of Value with given dimensions."""
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
