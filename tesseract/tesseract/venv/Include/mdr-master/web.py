import numpy as np
import matplotlib.pyplot as plt


class neuron:

    def _init_(self, input_neurons, weight=1, bias=-4):
        self.weight = weight
        self.bias = bias
        self.input = input_neurons
        self.output = 0
        self._output()

    def _output(self):
        try:
            if len(self.input) == 4:
                self.output = 0 if ((sum(self.input) + self.bias) * self.weight) < 0 else 1
            elif len(self.input) == 3:
                self.output = 0 if ((sum(self.input) + self.bias) * self.weight) < 0 else 1
        except:
            self.output = 0 if (self.weight * self.input + self.bias) < 0 else 1


output_matrix = np.zeros([81, 81], dtype=object)
masked = np.zeros([84, 84])

x_range = np.arange(-10, 10.25, 0.25)
y_range = np.arange(-10, 10.25, 0.25)
x_count = 0
y_count = 0

for x in x_range:
    # x only neurons
    # first layer
    n1 = neuron(x, bias=9)
    n2 = neuron(x, weight=-1, bias=-7)
    n3 = neuron(x, bias=3)
    n4 = neuron(x, weight=-1, bias=-1)
    n5 = neuron(x, bias=-2)
    n6 = neuron(x, weight=-1, bias=4)
    n7 = neuron(x, weight=-1, bias=6)
    n8 = neuron(x, weight=-1, bias=8)
    y_count = 0
    for y in y_range:
        # first layer
        n9 = neuron(y, bias=5)
        n10 = neuron(y, weight=-1, bias=-2)
        n11 = neuron(y, bias=1)
        n12 = neuron(y, weight=-1, bias=1)
        n13 = neuron(y, bias=-3)
        n14 = neuron(y, weight=-1, bias=5)
        # middle layer
        n15 = neuron([n1.output, n2.output, n9.output, n14.output])
        n16 = neuron([n3.output, n4.output, n9.output, n14.output])
        n17 = neuron([n1.output, n4.output, n9.output, n10.output])
        n18 = neuron([n5.output, n6.output, n9.output, n14.output])
        n19 = neuron([n5.output, n7.output, n11.output, n12.output])
        n20 = neuron([n5.output, n8.output, n13.output, n14.output])
        # output layer
        new_x = neuron([n15.output, n16.output, n17.output], bias=-1).output
        new_y = neuron([n18.output, n19.output, n20.output], bias=-1).output

        output_matrix[x_count, y_count] = [new_x, new_y]
        if new_x == 1:
            masked[x_count, y_count] = 1.0
        elif new_y == 1:
            masked[x_count, y_count] = 2.0
        else:
            masked[x_count, y_count] = 0.0
        y_count += 1
    x_count += 1

f = plt.figure(figsize=(12, 12))
for i in range(5):
    masked = np.rot90(masked)

plt.locator_params(axis='y', nbins=25)
plt.locator_params(axis='x', nbins=25)

plt.imshow(masked)