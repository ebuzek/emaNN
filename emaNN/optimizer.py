# base class
class Optimizer:
    
    def __init__(self, model):
       self.model = model

    def step(self):
      return None

    def zero_grad(self):
        self.model.zero_grad()
    

class SGD(Optimizer):
   
  def __init__(self, model, epsilon):
    super().__init__(model)
    self.model = model
    self.epsilon = epsilon

  def step(self):
    for p in self.model.parameters():
      p.data += -self.epsilon * p.grad

