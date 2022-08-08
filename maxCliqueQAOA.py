import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import networkx as nx
import csv

seeds = [42]

graphs_and_wires = []

# solution: 11100
edges = [(0, 1), (1, 2), (2, 0), (2, 3), (1, 4)]
graphs_and_wires.append((nx.Graph(edges), 5))

# # solution: 1110000
# edges = [(0, 1), (1, 2), (2, 0), (2, 3), (1, 4), (4, 5), (5, 2), (0, 6)]
# graphs_and_wires.append((nx.Graph(edges), 7))
#
# # solution: 1110010
# edges = [(0, 1), (1, 2), (2, 0), (2, 3), (1, 4), (4, 5), (5, 2), (0, 6), (0, 5), (1, 5)]
# graphs_and_wires.append((nx.Graph(edges), 7))
#
# # solution: 1110
# edges = [(0, 1), (0, 2), (0, 3), (1, 2)]
# graphs_and_wires.append((nx.Graph(edges), 4))
#
# # solution: 1111
# edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
# graphs_and_wires.append((nx.Graph(edges), 4))

for index in range(len(graphs_and_wires)):
    graph, n_wires = graphs_and_wires[index]

    for seed in seeds:

        np.random.seed(seed)

        cost_h, mixer_h = qml.qaoa.max_clique(graph, constrained=False)
        # print("cost_h:")
        # print(cost_h)
        # print("mixer_h:")
        # print(mixer_h)

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

            f = open('/home/jockel/master_2/QAOA/angles_graph_' + str(index) + '_seed_' + str(seed) + '_layers_' + str(
                n_layers) + '.csv', 'w')
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
            # TODO: optimizer = qml.GradientDescentOptimizer()
            opt = qml.GradientDescentOptimizer()

            # optimize parameters in objective
            params = init_params
            steps = 50
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


        _, bitstrings1, most_freq_1 = qaoa_maxclique(n_layers=1)
        _, bitstrings2, most_freq_2 = qaoa_maxclique(n_layers=2)
        _, bitstrings3, most_freq_3 = qaoa_maxclique(n_layers=3)
        _, bitstrings4, most_freq_4 = qaoa_maxclique(n_layers=4)
        _, bitstrings5, most_freq_5 = qaoa_maxclique(n_layers=5)
        _, bitstrings6, most_freq_6 = qaoa_maxclique(n_layers=6)

        xticks = range(0, pow(2, n_wires))
        print(xticks)
        xtick_labels = list(map(lambda x: x if x % (pow(2, n_wires-2) - 1) == 0 else None, xticks))
        print(xtick_labels)
        bins = np.arange(0, pow(2, n_wires) + 1) - 0.5
        print(bins)

        fig, (ax1) = plt.subplots(2, 3, figsize=(15, 10))
        plt.subplot(2, 3, 1)
        format_title = "n_layers={}, max={:0" + str(n_wires) + "b}"
        plt.title(format_title.format(1, most_freq_1))
        plt.xlabel("bitstrings")
        plt.ylabel("freq.")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist(bitstrings1, bins=bins)
        plt.subplot(2, 3, 2)
        plt.title(format_title.format(2, most_freq_2))
        plt.xlabel("bitstrings")
        plt.ylabel("freq.")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist(bitstrings2, bins=bins)
        plt.subplot(2, 3, 3)
        plt.title(format_title.format(3, most_freq_3))
        plt.xlabel("bitstrings")
        plt.ylabel("freq.")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist(bitstrings3, bins=bins)
        plt.subplot(2, 3, 4)
        plt.title(format_title.format(4, most_freq_4))
        plt.xlabel("bitstrings")
        plt.ylabel("freq.")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist(bitstrings4, bins=bins)
        plt.subplot(2, 3, 5)
        plt.title(format_title.format(5, most_freq_5))
        plt.xlabel("bitstrings")
        plt.ylabel("freq.")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist(bitstrings5, bins=bins)
        plt.subplot(2, 3, 6)
        plt.title(format_title.format(6, most_freq_6))
        plt.xlabel("bitstrings")
        plt.ylabel("freq.")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist(bitstrings6, bins=bins)
        plt.tight_layout()
        plt.savefig('/home/jockel/master_2/QAOA/results_graph_' + str(index) + '_seed_' + str(seed) + '.pdf')
