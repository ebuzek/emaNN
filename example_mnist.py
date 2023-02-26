# just to load mnist dataset
import random
from emaNN.optimizer import SGD
from emaNN.nn import MLP
from tensorflow.keras.datasets import mnist
import ssl

# ============
# model hyper parameters
HL_SIZE = 32 # hidden layer neuron count
# training
BATCH_SIZE = 64
EPOCHS = 20
STEP = 0.01
# ============

# utils
def get_one_hot(position, size):
  return [1 if position == i else 0 for i in range(size)]

def array1d(array2d):
  return [element for row in array2d for element in row]

# load dataset
print("loading MNIST dataset...")
ssl._create_default_https_context = ssl._create_unverified_context
(trainX, trainY), (testX, testY) = mnist.load_data()
print(f"Done. train size: {len(trainX)}, test size: {len(testX)}")

# create a Multi-Layer Perceptron
# input size: 28x28 = 784
# hidden layer(s) of size HL_SIZE
# output layer of size 10 (digits)
n = MLP(784, [HL_SIZE, HL_SIZE, 10])

print("Initialized MLP. Parameter count: ", len(n.parameters()))

# ===== TRAINING =======
optimizer = SGD(n, STEP)


def get_batch():
  i = random.randint(0, len(trainX)-BATCH_SIZE)
  return (trainX[i:i+BATCH_SIZE],trainY[i:i+BATCH_SIZE])

for e in range(EPOCHS):
  batchX, batchY = get_batch()
  # forarwd pass
  # print(f" ---- epoch {e}, target: {batchY[0]}")
  predictions = [n(array1d(x)) for x in batchX]
  targets = [get_one_hot(y, 10) for y in batchY]
  # loss = sum([(neuronOut - neuronY)**2 for neuronOut, neuronY in zip(t,p) for t, p in zip(targets, predictions)])
  loss = 0
  for (prediction, target) in zip(predictions, targets):
    for (neuronP, neuronT) in zip(prediction, target):
      loss += (neuronP - neuronT) ** 2

  print(f"epoch {e}: loss={loss.data}")

  # backward pass - compute gradients
  loss.backward()

  optimizer.step()
  optimizer.zero_grad()

## ===================
## TESTS
for i in range(10):
  tx, ty = (testX[i],testY[i])
  prediction = n(array1d(tx))
  print(f"[{ty}], prediction: ", [x.data for x in prediction])