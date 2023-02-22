import math

def wrap(other):
  return other if isinstance(other, Value) else Value(other, label=other)

# Scalar value node that is able to back-propagate gradient
class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0 
    self._prev = _children
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    other = wrap(other)
    out = Value(self.data + other.data, (self, other), '+')
    return out

  def __neg__(self, other):
    return self * (-1)

  def __sub__(self, other):
    return self + (-other)
  
  def __mul__(self, other):
    other = wrap(other)
    out = Value(self.data * other.data, (self, other), '*')
    return out

  def __radd__(self, other):
    return self + other

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    return self * other**-1

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')    
    return out

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self,), 'exp')
    return out

  def __pow__(self, other):
    out = Value(self.data ** other, (self,wrap(other)), 'pow', f'**{other}')
    return out

  def _backward(self):
    if not self._prev:
      return 
    if (self._op == '*'):
      self._prev[0].grad += self.grad * self._prev[1].data
      self._prev[1].grad += self.grad * self._prev[0].data
    elif self._op == '+':
      self._prev[0].grad += self.grad
      self._prev[1].grad += self.grad
    elif self._op == 'tanh':
      self._prev[0].grad += (1 - self.data**2) * self.grad
    elif self._op == 'exp':
      self._prev[0].grad += self.data * self.grad
    elif self._op == 'pow':
      self._prev[0].grad += self._prev[1].data * (self._prev[0].data**(self._prev[1].data - 1)) * self.grad
      self._prev[1].grad += self.data * self.grad
    else:
      raise Exception("")

  def backward(self):
    self.grad = 1
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()