import math, random

class Value:
    def __init__(self, data, children=(), operation=''):
        self.data = data
        self.previous = set(children)
        self.operation = operation
        self.gradient = 0.0
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.gradient += 1.0 * out.gradient
            other.gradient += 1.0 * out.gradient

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.gradient += other.data * out.gradient
            other.gradient += self.data * out.gradient

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "can only do int or float powers"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.gradient += other * (self.data ** (other-1)) * out.gradient

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")

        def _backward():
            self.gradient += (1 - t**2) * out.gradient

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.gradient += out.data * out.gradient

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(value):
            if value not in visited:
                visited.add(value)
                for child in value.previous:
                    build_topo(child)
                topo.append(value)
        build_topo(self)

        self.gradient = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()

        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


n = MLP(3, [4, 4, 1])

xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],]

ys = [1.0, -1.0, -1.0, 1.0]

for k in range(20):
    # forward
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward
    for p in n.parameters():
        p.gradient = 0.0

    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.1 * p.gradient

    # show epoch
    print(k, loss.data)
