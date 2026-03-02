"""Autograd engine."""
import math
import random


class Value:
    """Let there be Autograd to recursively apply the chain rule through a computation graph.

    Value wraps a single scalar number (.data) and tracks how it was computed.

    Think of each operation as a little lego block: it takes some inputs,
    produces an output (the forward pass), and it knows how its output would change with respect
    to each of its inputs (the local gradient).

    Thats all the information autograd needs from each block.
    Everything else is just the chain rule, stringing the blocks together.

    Every time you do math with Value objects (add, multiply, etc.),
    the result is a new Value that remembers its inputs (_children)
    and the local derivative of that operation (_local_grads).

    The full set of lego blocks:

    __add__
    __mul__
    __pow__
    log
    exp
    relu

    """

    __slots__ = ('data', 'grad', '_children', '_local_grads')  # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        """Make new instance."""
        self.data = data  # scalar value of this node calculated during forward pass
        self.grad = 0  # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children  # children of this node in the computation graph
        self._local_grads = local_grads  # local derivative of this node w.r.t. its children

    def __add__(self, other):
        """Define a + b. Local gradients: d/da = 1, d/db = 1."""
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        """Define a * b. Local gradients: d/da = b, d/db = a."""
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        """Define a ** n. Local gradients: d/da = n * a**(n-1)."""
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        """Define local gradients: d/da = 1/a."""
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        """Define local gradients: d/da = e**a."""
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        """Define local gradients: d/da = 1a>0."""
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
        """Walk this graph in reverse topological order.

        Starting from the loss, ending at the parameters, applying the chain rule at each step.

        After backward() completes, every Value in the graph has a .grad containing dL/dv,
        which tells us how the final loss would change if we nudged that value.
        """
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

        # We kick things off by setting self.grad = 1 at the loss node, because dL/dL=1
        # the losses rate of change with respect to itself is trivially 1.
        # From there, the chain rule just multiplies local gradients along every path back to the parameters.
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):  # pylint: disable=protected-access
                # Note the += (accumulation, not assignment).
                # When a value is used in multiple places in the graph (i.e. the graph branches),
                # gradients flow back along each branch independently and must be summed.
                # This is a consequence of the multivariable chain rule:
                # if c contributes to L through multiple paths, the total derivative is the sum
                # of contributions from each path.
                child.grad += local_grad * v.grad


def matrix(nout, nin, std=0.08):
    """Return matrix of Value with given dimensions."""
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def matrix2json(mat):
    """Return json data for given matrix."""
    return [[val.data for val in row] for row in mat]


def json2matrix(mat):
    """Return matrix for given json data."""
    return [[Value(val) for val in row] for row in mat]
