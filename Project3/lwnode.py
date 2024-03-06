class lwnode:
  def __init__(self, ncopy=None, par=None, nd=-1, pr=-1, cost=0):
    self.nd = nd
    self.pr = pr
    self.cost = cost

    if ncopy is not None:  # ncopy is lwnode
      self.nd = ncopy.nd
      self.pr = ncopy.pr
      self.cost = ncopy.cost

    if par is not None:  # par is lwnode
      self.par = par.nd
      self.cost += par.cost

  # define this for heap sorting:
  def __lt__(self, rhs):
    return self.cost < rhs.cost