import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from parameters import args
import constants as cs
import networkx as nx
import csv

np.random.seed(args.SEED)

graph = cs.GRAPHS[args.EXAMPLE_GRAPH]
n_wires = graph.size()

cost_h, mixer_h = qml.qaoa.max_clique(cs.GRAPHS[args.EXAMPLE_GRAPH],
                                      constrained=False)

shots = 1
dev = qml.device("default.qubit", wires=n_wires, shots=shots)


def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)


def bitarray_Z_to_int(bit_array_sample):
    s = bit_array_sample.T
    s = (1 - s.numpy()) / 2
    bit_string = "".join(str(int(bs)) for bs in s[0])
    return int(bit_string, base=2)


@qml.qnode(dev)
def circuit(gammas, betas, sample=False, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(n_layers):
        qml.templates.ApproxTimeEvolution(cost_h, gammas[i], 1)
        qml.templates.ApproxTimeEvolution(mixer_h, betas[i], 1)
    if sample:
        # measurement phase
        return [qml.sample(qml.PauliZ(i)) for i in range(n_wires)]
    # during the optimization phase we are evaluating a term
    # in the objective using expval
    return qml.expval(cost_h)


def qaoa_maxclique(n_layers=1):
    print("\np={:d}".format(n_layers))

    f = open('/home/jockel/master_2/QAOA/output/angles.csv', 'w')
    writer = csv.writer(f)
    row = ['step']
    for i in range(n_layers):
        row.append('gamma' + str(i))
        row.append('beta' + str(i))
    writer.writerow(row)

    # initialize the parameters near zero
    init_params = np.random.rand(2, n_layers)

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        obj = circuit(gammas, betas, sample=False, n_layers=n_layers)
        return obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.GradientDescentOptimizer(args.STEP_SIZE)

    # optimize parameters in objective
    params = init_params
    steps = args.NUM_STEPS
    for i in range(steps):
        params = opt.step(objective, params)
        row = [i]
        for l in range(n_layers):
            row.append(params[0][l])
            row.append(params[1][l])
        writer.writerow(row)
        if (i + 1) % 10 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

    # sample measured bitstrings 100 times
    bit_strings = []
    n_samples = 200
    for i in range(0, n_samples):
        bit_strings.append(bitarray_Z_to_int(circuit(params[0], params[1], sample=True, n_layers=n_layers)))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    formatstring = "Most frequently sampled bit string is: {:0" + str(n_wires) + "b}"
    print(formatstring.format(most_freq_bit_string))

    f.close()

    return -objective(params), bit_strings, most_freq_bit_string

_, bitstrings, most_freq = qaoa_maxclique(n_layers=args.LAYERS)

xticks = range(0, pow(2, n_wires))
xtick_labels = list(map(lambda x: x if x % (pow(2, n_wires - 2) - 1) == 0 else None, xticks))
bins = np.arange(0, pow(2, n_wires) + 1) - 0.5

fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
format_title = "n_layers={}, max={:0" + str(n_wires) + "b}"
plt.title(format_title.format(args.LAYERS, most_freq))
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings, bins=bins)
plt.tight_layout()
if args.WRITE:
    plt.savefig('/home/jockel/master_2/QAOA/output/result' + str(args.LAYERS) + '.pdf')
if args.SHOW:
    plt.show()
