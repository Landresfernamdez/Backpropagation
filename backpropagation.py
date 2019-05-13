import random
import math
import os
import sys


class NeuralNetwork:
    LEARNING_RATE = 0.5
    Final_Weights = []

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights=None,  output_layer_weights=None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden)
        self.output_layer = NeuronLayer(num_outputs)

        self.init_weights_from_inputs_to_hidden_layer_neurons(
            hidden_layer_weights)  # Se inicializan los pesos de la capa oculta
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(
            output_layer_weights)  # Se inicializan los pesos de la capa de salida

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(
                        random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(
                        hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(
                        random.random())
                else:
                    self.output_layer.neurons[o].weights.append(
                        output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(
            inputs)  # Calcula las salidas de la capa oculta
        # print("Salidas capa oculta:",hidden_layer_outputs)
        output_layer_outputs = self.output_layer.feed_forward(
            hidden_layer_outputs)  # Calcula las salidas de la capa de salida
        # print("salidas capa de salida",output_layer_outputs)
        return output_layer_outputs

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        
        self.feed_forward(training_inputs)
        # 1 Calcular el error de salida
        # Spk=(deseado - salida)*salida(1 - salida)
        errorescapasalida = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            errorescapasalida[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(
                training_outputs[o])
        # 2. Calculando el error de la capa oculta
        # salida(1-salida) * Sumatoria de (error de salida* peso de la salida)
        # Ypj(1-Ypj) * Σ Spk*Wkj
        errorescapaoculta = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            # Sumatoria de (error de salida* peso de la salida)
            # Σ Spk*Wkj
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += errorescapasalida[o] * \
                    self.output_layer.neurons[o].weights[h]
            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            #   Sumatoria de (error de salida* peso de la salida)* salida(1-salida)
            #   Σ(Spk*Wkj)*Ypj(1-Ypj)
            errorescapaoculta[h] = d_error_wrt_hidden_neuron_output * \
                self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input(
            )
        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                # Calculo del delta de capa salida
                # Δwkj (t+1)=Spk* Ypj
                # delta de capa oculta=Error de capa de oculta*entrada net
                pd_error_wrt_weight = self.LEARNING_RATE * \
                    errorescapasalida[o] * \
                    self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(
                        w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                # Actualizar pesos de capa salida
                # wkj(t+1)=wkj(t)+Δwji
                # wkj(t+1)=(wkj(t+1)-wkj(t))+Δwkj

                self.output_layer.neurons[o].weights[w_ho] = self.output_layer.neurons[o].weights[w_ho -
                                                                                                  1] + pd_error_wrt_weight
                print(self.output_layer.neurons[o].weights)
                self.Final_Weights = self.output_layer.neurons[o].weights
        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                # Calculo del delta de capa oculta
                # Δwji (t+1)=Spi* Xpi
                # delta de capa oculta=Error de capa de oculta*entrada net
                pd_error_wrt_weight = self.LEARNING_RATE * \
                    errorescapaoculta[h] * \
                    self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                # Actualizar pesos de capa oculta
                # wji(t+1)=wji(t)+Δwji
                # wji(t+1)=(wji(t+1)-wji(t))+Δwji
                self.hidden_layer.neurons[h].weights[w_ih] = self.hidden_layer.neurons[h].weights[w_ih -
                                                                                                  1] + pd_error_wrt_weight                

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(
                    training_outputs[o])
        return total_error


class NeuronLayer:
    def __init__(self, num_neurons):
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron())

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    def __init__(self):
        self.weights = []
    # Se encarga de calcular la salida de la neurona

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.operation(self.calculate_total_net_input())
        return self.output
    # Se encarga de calcular la net de la neurona

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total
    # Realiza la formula de la salida recibiendo la net como parametro
    # salida=1/1+e^-net

    def operation(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Error  de la salida
    # (deseado - salida)*salida(1 - salida)

    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input()
    # The error for each neuron is calculated by the Mean Square Error method:

    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2
    # Deseado - Salida
    # (dpk - ypk) si se quita el negativo de disminuye menos el error
    # -(dpk - ypk)

    def calculate_pd_error_wrt_output(self, target_output):
        return (target_output - self.output)
    # Salida * (1- Salida)

    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)
    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ

    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


'''nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], output_layer_weights=[0.4, 0.45, 0.5, 0.55])
for i in range(10000):
    nn.train([0.05, 0.1], [0.01, 0.99])
    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

 XOR example:'''

training_sets = []


def cargarTrainingSet():
    lista = os.listdir(os.path.dirname(
        os.path.abspath(__file__)) + "\\inputs\\")
    for filename in lista:
        trainsetElement = []
        currentPath = os.path.dirname(os.path.abspath(__file__)) + "\\inputs\\"
        index = lista.index(filename)
        outputNN = []
        inputNN = []
        if(index % 2 == 0):
            outputNN.append(0)
        else:
            outputNN.append(1)
        with open(currentPath + filename, "r") as f:
            linea = f.read()
            for elemento in linea:
                if(elemento == '1' or elemento == '0'):
                    binary = int(elemento)
                    inputNN.append(binary)
        trainsetElement.append(inputNN)
        trainsetElement.append(outputNN)
        training_sets.append(trainsetElement)


training_setsprueba = []


def cargarTrainingPrueba():
    lista = os.listdir(os.path.dirname(
        os.path.abspath(__file__)) + "\\inputsdesconocidas\\")
    for filename in lista:
        trainsetElement = []
        currentPath = os.path.dirname(os.path.abspath(
            __file__)) + "\\inputsdesconocidas\\"
        index = lista.index(filename)
        outputNN = []
        inputNN = []
        if(index % 2 == 0):
            outputNN.append(0)
        else:
            outputNN.append(1)
        with open(currentPath + filename, "r") as f:
            linea = f.read()
            for elemento in linea:
                if(elemento == '1' or elemento == '0'):
                    binary = int(elemento)
                    inputNN.append(binary)
        trainsetElement.append(inputNN)
        trainsetElement.append(outputNN)
        training_setsprueba.append(trainsetElement)


if __name__ == "__main__":
    cargarTrainingSet()
    nn = NeuralNetwork(len(training_sets[0][0]), 2, len(training_sets[0][1]))
    for i in range(1000):
        training_inputs, training_outputs = random.choice(training_sets)
        nn.train(training_inputs, training_outputs)

        print(i, nn.calculate_total_error(training_sets))
    print("capa salida Salida:", nn.output_layer.neurons[0].output)
    print("capa oculta Salida:", nn.hidden_layer.neurons[0].output)

    print("Prueba")
    cargarTrainingPrueba()
    training_inputs_d, training_outputs_d = random.choice(training_setsprueba)
    nn.feed_forward(training_inputs_d)

    print("capa salida Salida:", nn.output_layer.neurons[0].output)
    print("capa oculta Salida:", nn.hidden_layer.neurons[0].output)
    print("Pesos finales:", nn.Final_Weights)
