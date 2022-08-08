import numpy as np
import argparse
from argparse import RawTextHelpFormatter


def tuple_type(s):
    try:
        s = s.replace("(", "").replace(")", "")
        lower, upper = map(int, s.split(','))
        return range(lower, upper)
    except:
        raise argparse.ArgumentTypeError(
            'LAYERS must be of the form "lower,upper" or "(lower,upper)"')

def graph_type(s):
    try:
        edge_strings = s.split('; ')
        edges = []
        for edge in edge_strings:
            edges.append(edge_type(edge))
        return edges
    except:
        raise argparse.ArgumentTypeError(
            'GRAPH must be of the form "EDGE; EDGE; ...")')


def edge_type(s):
    try:
        s = s.replace("(", "").replace(")", "")
        a, b = map(int, s.split(','))
        return a, b
    except:
        raise argparse.ArgumentTypeError(
            'EDGE must be of the form "vertex,vertex" or "(vertex,vertex)"')


parser = argparse.ArgumentParser(
    description='A quantum approach to approximate the maximum Clique of a given graph.',
    formatter_class=RawTextHelpFormatter)
parser.add_argument('--examplegraph', dest='EXAMPLE_GRAPH', type=int, default=0, choices=[0, 1, 2, 3, 4],
                    help='Use an example graph for the approximation.\nThere are 5 example graphs:\n'
                         'graph 0: (0, 1), (1, 2), (2, 0), (2, 3), (1, 4), optimal solution: 11100\n'
                         'graph 1: (0, 1), (1, 2), (2, 0), (2, 3), (1, 4), (4, 5), (5, 2), (0, 6), optimal solution: '
                         '1110000\n'
                         'graph 2: (0, 1), (1, 2), (2, 0), (2, 3), (1, 4), (4, 5), (5, 2), (0, 6), (0, 5), (1, 5), '
                         'optimal solution: 1110010\n'
                         'graph 3: (0, 1), (0, 2), (0, 3), (1, 2), optimal solution: 1110\n'
                         'graph 4: (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), optimal solution: 1111')
parser.add_argument('--layers', dest='LAYERS', type=int, default=5,
                    help='Number of Layers to be used for the approximation')
parser.add_argument('--show', dest='SHOW', action='store_const', const=False, default=True,
                    help='Show the output graph immediately')
parser.add_argument('--write', dest='WRITE', action='store_const', const=False, default=True,
                    help='Write the graph to a pdf file')
parser.add_argument('--steps', dest='NUM_STEPS', type=int, default=50,
                    help='Number of time steps to simulate')
parser.add_argument('--stepsize', dest='STEP_SIZE', type=float, default=1.,
                    help='Size of one time step.')
parser.add_argument('--seed', dest='SEED', type=int, default=42, help='Random seed to set the initial values of all '
                                                                      'gamma and beta parameters')
parser.add_argument('--optimizer', dest='OPTIMIZER', nargs='*', default='GradientDescentOptimizer',
                    choices=['GradientDescentOptimizer', 'AdagradOptimizer', 'AdamOptimizer'],
                    help='Optimizer used to optimize gamma and beta')

args = parser.parse_args()
