import numpy as np
from parameters import args
import networkx as nx

GRAPHS = [nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (1, 4)]),
          nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (1, 4), (4, 5), (5, 2), (0, 6)]),
          nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (1, 4), (4, 5), (5, 2), (0, 6), (0, 5), (1, 5)]),
          nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2)]), nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])]

