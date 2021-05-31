# Perceptron-Multicapa
Implementación básica de una red neuronal, el cual será usado para predecir si una canción le gusta a una persona o no.

## Definición de la neurona
Se define la estructura de la neurona, la cual contendrá toda la información necesaria para que pueda realizar las operaciones que requiere. Se encuentra el detalle completo en el siguiente [archivo](/src/notebooks/NeuronalNetwork.ipynb)

    class Neuron:
      def __init__(self, function, value=0.0, id=None):
        self.value = value
        self.id = id
        self.funct = function
        self.der_funct = lambda F_x: F_x * (1 - F_x)
        self.next_layer = []
        self.prev_layer = []
        self.weights = []

### Métodos
El método encargado para agregar capas a las neuronas

      def add_layer(self, nodes_layer=None):

El método encargado de calcular y retornar el valor de la neurona.

      def calculate_value(self, apply_function=True):

### Probando la neurona
Se realizan algunas pruebas para verificar que todas las funciones las realiza de una forma correcta. Se crea una neurona que use la función sigmoide como función de activación, luego se le agrega una capa a partir de los valores que deberian tener los nodos. Por ultimo se calcula el valor activado de la neurona.

    import math

    def sigmoide(x):
      return 1 / (1 + math.exp(-x))

    values = [0.34, 0.99, 0.03]
    nodes = [Neuron(sigmoide, v) for v in values]

    n = Neuron(sigmoide)
    n.add_layer(nodes)

    n.calculate_value()
    
Obteniendo el valor de 

    0.550154918047423
    
    
## Definición de la red neuronal
Se defina la estructura de la red neuronal, la cual tendra las funciones básicas para que pueda operar.

    class NeuronalNetwork:
      def __init__(self, cant_input, cant_output, function):
        self.input_layer = [Neuron(function, id=(0, i + 1)) for i in range(cant_input)]
        self.output_layer = [Neuron(function, id=(-1, i + 1)) for i in range(cant_output)]
        self.hidden_layers = []
        
### Métodos

El método encargado de agregar las capas a la red neuronal, obteniendo la malla, conectandola con la capa de entrada y finalmente con la capa de salida

    def add_layers(self, layers):

Para crear las conexiones entre las capas ingresadas, se utilizan las funciones de a continuación, en donde la primera sólo crea una malla de capas, pero sin conexiones entre ellas. La segunda es la encargada de generar la conexión nodo a nodo.

    def _create_mesh(self, layers):

    def _make_connections(self, layers):
      
Para realizar las predicciones se copian los valores del input en los nodos de la capa de entrada, se calcula los valores de las capas ocultas y se calcula la salida sin aplicar la funcion de activación al valor.

    def predict(self, input):
    
Luego para cada se agregan todas las predicciones a una lista y se retornan al main con:
    
    def predictions(self, X_train):
          
Para entrenar el modelo, se genera la prediccion de los valores, se calcula la diferencia (delta) con el valor esperado y se actualizan los pesos según corresponda.

    def fit(self, X, y, learning_rate=0.1):
    
Entre otros metodos necesarios para que la red funcione se tienen los siguientes:

* Función encargada de actualizar los pesos según los deltas

      def _update_weights(self, deltas, learning_rate):
    
* Función que actualiza los pesos de la capa según los deltas

      def _update_layer_weight(self, layer, i_delta, deltas, learning_rate):
    
* Función que calcula los deltas de la capa de salida

      def _calculate_deltas(self, y):
    
* Función que inicializa la lista de deltas para cada capa (oculta y de salida)
  
      def _create_delta_set(self):
      
## Implementación de la red

Lo primero que se debe hacer antes de empezar a implementar la red es la obtención y limpieza de datos. Para ello se trabajo el [dataset](/src/dataset/data.csv) con pandas para obtener el vector de valores a ingresar en cada neurona. Puede ver el detalle en el siguiente [archivo](/src/notebooks/Dataset.ipynb)

### Obtención y limpieza de datos

Del dataset original, se utilizaron los campos numéricos, se muestra un ejemplo con los primeros 5 registros

    acousticness	danceability	duration_ms	energy	instrumentalness	key	liveness	loudness	mode	speechiness	  tempo	time_signature	valence
          0.0102	       0.833	     204600	 0.434	        0.021900	  2	  0.1650	  -8.795	  1	       0.4310	150.062	           4.0	  0.286	
          0.1990	       0.743     	 326933	 0.359	        0.006110	  1	  0.1370	  -10.401	  1	       0.0794	160.083	           4.0	  0.588
          0.0344	       0.838	     185707	 0.412	        0.000234	  2	  0.1590	  -7.148	  1	       0.2890	75.044	           4.0	  0.173
          0.6040	       0.494	     199413	 0.338	        0.510000	  5	  0.0922	  -15.236	  1	       0.0261	86.468	           4.0	  0.230
          0.1800	       0.678	     392893	 0.561	        0.512000	  5	  0.4390	  -11.648	  0	       0.0694	174.004	           4.0	  0.904

### División de datos de entrenamiento y prueba

Para realizar la separación de datos se implementó la siguiente función, la cual recibe los datos de entrada (X,y) y separa de manera aleatoria un 70% datos de entrenamiento y un 30% de datos para su posterior prueba. Finalmente retorna una tupla con 4 datos: Los valores de X e y de entrenamiento y testeo

    import random

    def get_values_dataset():
        df = pd.read_csv('data.csv', index_col=0)

        # Crear variables
        X = df.drop(labels=['song_title', 'artist', 'target'], axis=1)
        y = df['target']

        cols_to_normalize = ['duration_ms', 'key', 'tempo', 'time_signature']

        # Normalizar el conjunto de entrada
        for col in cols_to_normalize:
          X[col] = df[col] / df[col].max()

        # Se normaliza por el min al tener valores negativos.
        X.loudness = X.loudness / X.loudness.min()

        return (X.values.tolist(), y.values.tolist())

            return (X_train, X_test, y_train, y_test)
        
### Modelo de entrenamiento
Se crea la red neuronal, se agregan las capas y se inicializan los datos.

    layers = [2, 3]

    nn = NeuronalNetwork(size_X, size_y, sigmoide)
    nn.add_layers(layers)
    
Se entrena el modelo

    nn.fit(X_train, y_train)
    
### Modelo de pruebas
Se generan las predicciones y se almacenan en una lista para comparar con las teorícas (y_test)

    predictions = nn.predictions(X_test)
    
## Conclusiones
Si comparamos los resultados generados en las predicciones de nuestro perceptrón, obtenemos una precisión del 50% de los datos.
Haciendo un analisis de los resultados, podemos ver que los datos de las predicciones son muy similares, y haciendo el ruteo de
la red podemos ver como se van ajustando los pesos. Finalmente, nuestra red no es capaz de predecir correctamente si al usuario
le gusta una canción con los datos extraidos desde el dataset a pesar de intentar con distintas capas y tamaños de testeo.

## Video
En el siguiente [link](https://drive.google.com/drive/folders/18j8_XpWpYjkRARHtlxg8IixK0LkxsiCk?usp=sharing) se encuentra el video explicativo del desafío.
