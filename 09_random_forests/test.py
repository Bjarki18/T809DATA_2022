import numpy as np

# feat = np.linspace(0, 100, 20)
# tree = np.linspace(0, 200, 40)

feats = [i for i in range(0,105,5)]
feats[0] = 1


trees = [i for i in range(0,205,10)]
trees[0] = 1


print(len(feats) * len(trees))


