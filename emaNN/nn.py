import random
from emaNN.value import Value

# base class
class Module:

  def parameters(self):
    return None

  # set params to 0
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0

class Neuron:

  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    # tanh( w * x + b )
    act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
    # out = act.relu()
    # out = act.tanh()
    out = act
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
    return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):

  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]