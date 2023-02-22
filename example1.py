

from emaNN.nn import MLP
from emaNN.optimizer import SGD

EPOCHS = 20
STEP = 0.05
def train_sgd(n, examples, targets):
  print("target:", targets)
  print("initial predictions: ", [n(x) for x in examples])
  optimizer = SGD(n, STEP)
  for e in range(EPOCHS):
    # forarwd pass
    predictions = [n(x) for x in examples]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(targets, predictions)])
    # backward pass - compute gradients
    loss.backward()

    print(f"epoch {e}: loss={loss.data}")

    optimizer.step()
    optimizer.zero_grad()
    
  final = [n(x) for x in examples]
  print("final predictions: ", final)

# create a Multi-Layer Perceptron
# input size: 3
# 3 hidden layers of sizes [4, 4, 1]
n = MLP(3, [4, 4, 1])

print("> MLP. Total number of parameters: ", len(n.parameters()))

# exmaple inputs
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, .5],
    [.5, 1., 1.],
    [1.0, 1.0, -1.0]
]
# expected targets
ys = [1., -1., -1., 1]

# predictions = [n(x) for x in xs]
# loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, predictions)])
# draw_dot(loss)

train_sgd(n, xs, ys)
