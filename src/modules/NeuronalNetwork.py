import random
import math

class Neuron:
  def __init__(self, function, value=0.0, id=None):
    self.value = value
    self.id = id
    self.funct = function
    self.der_funct = lambda F_x: F_x * (1 - F_x)
    self.next_layer = []
    self.prev_layer = []
    self.weights = []
    self.bias = 0 # no yet implemented


  def add_layer(self, nodes_layer=None):
    self.prev_layer = nodes_layer
    
    for node in nodes_layer:
      node.next_layer.append(self)
      
    self.weights = [random.random() for n in nodes_layer]


  def calculate_value(self, apply_function=True):
    sum = 0.0
    for (index, node) in enumerate(self.prev_layer):
      sum += node.value * self.weights[index]

    self.value = sum + self.bias

    if apply_function:
      self.value = self.funct(self.value)

    return self.value


def sigmoide(x):
  return 1 / (1 + math.exp(-x))

def print_connection_info(verbose_level, node, prev, index_weight):
  if 1 == verbose_level:
    print(f'-> {prev.id}')
  elif 2 == verbose_level:
    print(f'w: {node.weights[index_weight]} ====>>   {prev.id}')
  elif 3 == verbose_level or 4 == verbose_level:
    print(f'w: {node.weights[index_weight]} ====>>   {prev.id}: ({prev.value})')

def print_node_info(verbose_level, node):
  if 1 == verbose_level or 2 == verbose_level:
    print(f'{node.id}')
  elif 3 == verbose_level:
    print(f'{node.id}: ({node.value})')
  elif 4 == verbose_level:
    print(f"{node.id}: f(x)=({node.value}) f'(x)=({node.der_funct(node.value)})")

def show_neuronal_network(network, verbose_level=1):
  print('* Input layer nodes have "0, x" by id')
  print('* Output layer nodes have "-1, x" by id')
  print('* Hidden layers nodes have "n, x" by id, where n represents the length of layer\n')

  for (i, layer) in enumerate(network.hidden_layers):
    print('Hidden Layer: ', i + 1)

    for node in layer:
      print_node_info(verbose_level, node)

      for (k, prev) in enumerate(node.prev_layer):
        print_connection_info(verbose_level, node, prev, k)

      print()

  print('Output Layer')
  for node in network.output_layer:
    print_node_info(verbose_level, node)

    for (j, prev) in enumerate(node.prev_layer):
      print_connection_info(verbose_level, node, prev, j)

    print()

def show_predict_info(network, x, verbose_level=1):
  y = nn.predict(x)

  print(f'Value to predict: {x}')
  print(f'Predicted value: {y}\n')
  print('Network status information after predicting: \n')

  show_neuronal_network(nn, verbose_level)

class NeuronalNetwork:
  def __init__(self, cant_input, cant_output, function):
    self.input_layer = [Neuron(function, id=(0, i + 1)) for i in range(cant_input)]
    self.output_layer = [Neuron(function, id=(-1, i + 1)) for i in range(cant_output)]
    self.hidden_layers = []

  def is_empty(self):
    return len(self.hidden_layers) == 0

  def get_function(self):
    return self.input_layer[0].funct

  def add_layers(self, layers):
    # Se obtiene la malla
    self.hidden_layers = self._make_connections(layers)

   # Se conecta el principio de la malla con la capa de entrada
    for node in self.hidden_layers[0]:
      node.add_layer(nodes_layer=self.input_layer)

    # Se conecta la capa de salida con el final de la malla
    for node in self.output_layer:
      node.add_layer(nodes_layer=self.hidden_layers[-1])


  def fit(self, X, y, learning_rate=0.1):
    # Por cada vector del conjunto de inputs, se ajusta el modelo
    for (i, x) in enumerate(X):
      print('\n------------------------------------------------------------------')
      print('Current fit iteration:', i)
      print(f'Expected value at this current fit iteration: {y[i]}')

      self.predict(x)

      print('\nNetwork Information: \n')
      show_predict_info(self, x, 4)

      deltas = self._calculate_deltas(y[i])

      print('Deltas information: ')
      print('Output layer deltas: ')
      for (i, node) in enumerate(self.output_layer):
          print(f'\t{node.id}: {deltas[-1][i]}')

      print('\nHidden layers deltas: ')
      for (i, hidden_deltas) in enumerate(deltas[:-1]):
        print('\nLayer:', i)
        
        for (j, node) in enumerate(self.hidden_layers[i]):
          print(f'\t{node.id}: {hidden_deltas[j]}')

      self._update_weights(deltas, learning_rate)

      print('\nNetwork Information after update the weights: \n')
      show_neuronal_network(self, 2)

  
  def predict(self, input):
    # Se copian los valores del input en los nodos de la capa de entrada
    for (index, value) in enumerate(input):
      self.input_layer[index].value = value

    # Se calculan los valores de las capas ocultas
    for layer in self.hidden_layers:
      for node in layer:
        node.calculate_value()

    # Se calcula la salida sin aplicar la funcion de activacion al valor
    for node in self.output_layer:
      node.calculate_value()

    return [node.value for node in self.output_layer]

  def predictions(self, X_train):
    predictions = []
    # Se predice la salida para cada dato de entrenamiento y se retorna en formato lista
    for x in X_train:
      predictions.append( self.predict(x) )

    return predictions


  def _make_connections(self, layers):
    '''
    Realiza las conexiones entre las capas ingresadas, y retorna una malla de 
    la red neuronal (sin entrada ni salida)
    '''
    nodes_mesh = self._create_mesh(layers)
    length = len(layers)

    for i in range(1, length):
      for node in nodes_mesh[i]:
        node.add_layer(nodes_layer=nodes_mesh[i - 1])

    return nodes_mesh


  def _create_mesh(self, layers):
    '''
    Crea una malla de capas, pero sin conecciones entre ellas
    '''
    nodes_mesh = []
    function = self.get_function()

    for current_layer in layers:
      new_nodes_layer = [Neuron(function, id=(current_layer, j + 1)) for j in range(current_layer)] # for debug
      #new_nodes_layer = [Neuron(function) for j in range(current_layer)]
      nodes_mesh.append(new_nodes_layer)

    return nodes_mesh


  def _update_weights(self, deltas, learning_rate):
    # Se actualizan los pesos de la primera capa oculta
    self._update_layer_weight(self.input_layer, 0, deltas, learning_rate)

    # Se actualizan los pesos de la segunda capa oculta
    for (i, layer) in enumerate(self.hidden_layers):
      self._update_layer_weight(layer, i + 1, deltas, learning_rate)


  def _update_layer_weight(self, layer, i_delta, deltas, learning_rate):
    for (i, node) in enumerate(layer):
      for (j, next) in enumerate(node.next_layer):
        new_weight = -learning_rate * deltas[i_delta][j] * next.value
        next.weights[i] = new_weight


  def _calculate_deltas(self, y):
    deltas = self._create_delta_set()
    
    # Se calculan los deltas de la capa de salida
    for (index, node) in enumerate(self.output_layer):
      deltas[-1][index] = (node.value - y[index]) * node.der_funct(node.value)

    length_hidden = len(self.hidden_layers)

    # Se calculan los deltas de las capas ocultas
    for i in range(length_hidden):
      i_lay = length_hidden - i - 1
      layer = self.hidden_layers[i_lay]

      for (j, node) in enumerate(layer):
        sum = 0.0

        for (k, next) in enumerate(node.next_layer):
          sum += deltas[i_lay + 1][k] * next.weights[j]

        deltas[i_lay][j] = sum * node.der_funct(node.value)

    return deltas

  def _create_delta_set(self):
    deltas_hidden = [[0.0 for v in layer] for layer in self.hidden_layers]
    deltas_output = [0.0 for v in self.output_layer]

    return [*deltas_hidden, deltas_output]

def train_test_split(X, y, test_size=0.3):
    # Inicialización de listas
    X_train, X_test, y_train, y_test = (list(),list(),list(),list())
    # Se define tamaño de entrenamiento (70%)
    size_train = round(len(X)*(1 - test_size))
    # Se genera una lista con los indices del 70% de datos aleatorios.
    random_range = random.sample(range(len(X)), size_train)

    for index in range(len(X)):
        if index in random_range:
            X_train.append( X[index] )
            y_train.append( y[index] )
        else:
            X_test.append( X[index] )
            y_test.append( y[index] )
    
    return (X_train, X_test, y_train, y_test)