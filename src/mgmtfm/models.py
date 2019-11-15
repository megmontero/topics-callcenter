import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, GRU, LSTM
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras import regularizers
from sklearn.metrics import confusion_matrix
import keras



class Models:
    """
    Clase para implementar diferentes modelos orientados a PLN.
    """
    
    _model = None
    _nclasses = 0
    _sequence_lenght = 0
    _embedding_dim = 0
    _vocabulary_size = 0
    _embedding_matrix = None
    _conf_optuna = None
    optuna = None
    
    def __init__(self, nclasses=1, sequence_length=None, embedding_dim=None, vocabulary_size=None, embedding_matrix=None, load=False, path=None):
        """
        Constructor de la clase. 
        
        Args: 
            nclasses (int): Número de clases que queremos extraer con nuestro modelo.

            sequence_length (int): Longitud de la secuencia.

            embedding_dim (int): Dimensión del embedding usado.

            vocabulary_size (int): Tamaño del vocabulario. 

            embedding_matrix (matriz): Matriz de embedding.

            load (bool): Si es necesario cargar un modelo previo. 

            path (str): En el caso de querer cargar un modelo previo. El path.
        """
        if (load ==True):
            self.load_model(path)
        else:
            self._nclasses = nclasses
            self._sequence_lenght = sequence_length
            self._embedding_dim = embedding_dim
            self._vocabulary_size = vocabulary_size 
            self._embedding_matrix = embedding_matrix
    
    #esblecemos el valor de dropout y los filtros (que determinaran tambien el número de capas concolucionales )
    def model_cnn_1(self, filter_sizes=[3,4,5], drop=0.3, num_filters = 100, regl2=0.01):
        """
        Crear un modelo de redes neuronales convolucional con diferentes capas. 
        
        Args: 
            filter_sizes (list): Lista con los tamaños de los kernel de las capas convolucionales. 
            La longitud de la lista determinará las capas convolucionales de la red.

            drop (float): Dropout.

            num_filters (int): Número de filtros a usar en cada capa.

            regl2 (float): Regularizador de pesos nivel 2.
        """
    
        
        embedding_layer = Embedding(self._vocabulary_size,
                            self._embedding_dim,
                            weights=[self._embedding_matrix],
                            trainable=False)
        
        inputs = Input(shape=(self._sequence_lenght,))
        embedding = embedding_layer(inputs)
        reshape = Reshape((self._sequence_lenght,self._embedding_dim,1))(embedding)
        
        cnn_layers = []
        for filter_size in filter_sizes:
            conv = Conv2D(num_filters, (filter_size, self._embedding_dim),activation='relu',kernel_regularizer=regularizers.l2(regl2))(reshape)
            maxpool = MaxPooling2D((self._sequence_lenght - filter_size + 1, 1), strides=(1,1))(conv)
            cnn_layers.append(maxpool)

        merged_tensor = concatenate(cnn_layers, axis=1)
        flatten = Flatten()(merged_tensor)
        reshape = Reshape((len(filter_sizes)*num_filters,))(flatten)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=self._nclasses, activation='softmax',kernel_regularizer=regularizers.l2(regl2))(dropout)

        self._model = Model(inputs, output)
        
        
    def model_dense_1(self,sizes=[500, 300, 200, 150, 100, 50, 20, 1],drop=0.2):  
        """
        Crear un modelo de redes neuronales con capas totalmente conectadas.
        
        Args: 
            sizes (list): Lista con los tamaños de las capas. 
            La longitud de la lista determinará las capas de la red.
            La primera capa debe coincidir con el tamaño del input y la última 
            con la salida (si es 1 será una clasificación binaria). 

            drop (float): Dropout.
        """
        
        self._model = Sequential()
        self._model.add(Dense(sizes[0], activation='relu', name="Input", input_shape=(sizes[0],)))
        for i in range(1, len(sizes) -1):
            self._model.add(Dense(sizes[i], activation='relu', name="Hidden_layer_" + str(i)))
            self._model.add(Dropout(drop))
        if sizes[-1] == 1:
            self._model.add(Dense(1, activation='sigmoid', name="Output_layer" ))
        else:
            self._model.add(Dense(sizes[-1], activation='softmax', name="Output_layer" ))
        return self._model
        
        
    def model_cnn_1_bin(self, filter_sizes=[3,4,5], drop=0.3, num_filters = 100, regl2=0.01):
        """
        Crear un modelo de redes neuronales convolucional con diferentes capas para clasificación 
        binaria.
        
        Args: 
            filter_sizes (list): Lista con los tamaños de los kernel de las capas convolucionales. 
            La longitud de la lista determinará las capas convolucionales de la red.

            drop (float): Dropout.

            num_filters (int): Número de filtros a usar en cada capa.

            regl2 (float): Regularizador de pesos nivel 2.
        """
        embedding_layer = Embedding(self._vocabulary_size,
                            self._embedding_dim,
                            weights=[self._embedding_matrix],
                            trainable=False)
        
        inputs = Input(shape=(self._sequence_lenght,))
        embedding = embedding_layer(inputs)
        reshape = Reshape((self._sequence_lenght,self._embedding_dim,1))(embedding)
        
        cnn_layers = []
        for filter_size in filter_sizes:
            conv = Conv2D(num_filters, (filter_size, self._embedding_dim),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
            maxpool = MaxPooling2D((self._sequence_lenght - filter_size + 1, 1), strides=(1,1))(conv)
            cnn_layers.append(maxpool)

        merged_tensor = concatenate(cnn_layers, axis=1)
        flatten = Flatten()(merged_tensor)
        reshape = Reshape((len(filter_sizes)*num_filters,))(flatten)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=1, activation='sigmoid',kernel_regularizer=regularizers.l2(regl2))(dropout)

        self._model = Model(inputs, output)
        
        
    def summary_model(self):
        """
        Presentar un resumen del modelo actual.
        """
        if (self._model):
            self._model.summary()
        else:
            print("Model not initialized")
            
    def compile_and_train(self, X_train, y_train, batch_size=50, epochs=10, verbose=1, 
                          lr=1e-3, decay=1e-6,validation_data=None,metrics=["acc"], callbacks=None, loss='categorical_crossentropy'):
        """
        Compilar y entrenar el modelo definido. 
        
        Args:
            X_train (array): Conjunto de entrenamiento.

            y_train (array): Etiquetas de entrenamiento. 

            batch_size (int): Tamaño de batch para el entrenamiento.

            epochs (int): Número de epochs para el entrenamiento.

            verbose (int): Tipo de verbose usado en el entrenamiento. 

            lr (float): Tasa de aprendizaje del entrenamiento. 

            decay (float): Pesos decayentes para las redes neuronales.      
        """
        adam = Adam(lr, decay=decay)
        
        self._model.compile(loss=loss,
                        optimizer=adam,
                        metrics=metrics  )
        self._model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                        validation_data=validation_data, callbacks=callbacks) 
        
    def get_confusion_matrix(self, X_test, y_test, binary=False):
        """
        Obtener la matriz de confusión. 
         Args:
            X_test (array): Conjunto de test.

            y_test (array): Etiquetas de test. 

            binary (bool): Si se trata de una clasificación binaria.       
        """
        y_pred = self._model.predict(X_test)
        if (binary):
            y_pred = [1 * (x[0]>=0.5) for x in y_pred]
            y_true = y_test
        else:
            y_pred = [1 * (x[0]>=0.5) for x in y_pred]
            y_true = [1 * (x[0]>=0.5) for x in y_test]
        return confusion_matrix(y_true, y_pred)
    
    
    def model_rnn_1(self,  memory_units=200, cell=1, drop=0.2):
        """
        Crear un modelo de redes neuronales recurrentes. 
        
        Args: 
            memory_units (int): Número de celdas. 

            cell (int): Tipo de celda. 0-> LSTM 1->GRU

            drop (float): Dropout
        
        """
        self._model = Sequential()
        embedding_layer = Embedding(self._vocabulary_size,
                            self._embedding_dim,
                            weights=[self._embedding_matrix],
                            trainable=False)
    
        self._model.add(embedding_layer)
        self._model.add(Dropout(drop))
        if cell==0:
            self._model.add(LSTM(memory_units))
        elif cell==1:
            self._model.add(GRU(memory_units))
        self._model.add(Dropout(drop))
        if self._nclasses == 1:
            self._model.add(Dense(1, activation='sigmoid', name="Output_layer" ))
        else:
            self._model.add(Dense(self._nclasses, activation='softmax', name="Output_layer" ))

    def model_rnn_2(self,  memory_units=200, cell=1, drop=0.2, filters=32, kernel_size=3, pool_size=2, padding="same"): 
        """
        Crear un modelo combinando redes neuronales recurrentes con una capa convolucional.
        
        Args: 
            memory_units (int): Número de celdas. 

            cell (int): Tipo de celda. 0-> LSTM 1->GRU

            drop (float): Dropout

            filters (int): Número de filtros de la capa convolucional. 

            kernel_size (int): Tamaño de kernel de la capa convolucional.

            pool_size (int): Tamaño del poolin. 

            padding (str): Tipo de padding a realizar. 
        
        """
        self._model = Sequential()
        embedding_layer = Embedding(self._vocabulary_size,
                            self._embedding_dim,
                            weights=[self._embedding_matrix],
                            trainable=False)
    
        self._model.add(embedding_layer)
        self._model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation='relu'))
        self._model.add(MaxPooling1D(pool_size=pool_size))
        if cell==0:
            self._model.add(LSTM(memory_units))
        elif cell==1:
            self._model.add(GRU(memory_units))
        self._model.add(Dropout(drop))
        if self._nclasses == 1:
            self._model.add(Dense(1, activation='sigmoid', name="Output_layer" ))
        else:
            self._model.add(Dense(self._nclasses, activation='softmax', name="Output_layer" ))
            
            
    def get_best_model(self, name, params):
        """
        Devolver un modelo a partir del nombre y los parámetros. 
        
        Args: 
            name (str): Nombre del modelo.
            
            params (dict): Parámetros del modelo.
        """
        name = name.lower()
        
        if "dense_" in name:
            self.model_dense_1(drop=params["dropout"])
        elif "cnn_" in name:
            filter_sizes = [params["kernel_size_conv_1"],params["kernel_size_conv_2"],params["kernel_size_conv_3"] ]
            if (self._nclasses == 1):
                self.model_cnn_1_bin(filter_sizes=filter_sizes, drop=params["dropout"], 
                                 num_filters=params["num_filters"] )
            else: 
                self.model_cnn_1(filter_sizes=filter_sizes, drop=params["dropout"], 
                                 num_filters=params["num_filters"] )
        elif "rnn_" in name:
            self.model_rnn_1(params["memory_units"], params["cell"], params["dropout"])
        elif "rnn2_" in name:
            self.model_rnn_2(params["memory_units"], params["cell"], params["dropout"], params["filters"], 
                             params["kernel_size"], params["pool_size"], params["padding"])
        

    
    def save_model(self, path):
        """
        Guardar un modelo. 
        
        Args:
            path (str): Ruta donde guardar el modelo.
        """
        self._model.save(path) 
        
    def load_model(self, path):
        """
        Cargar un modelo. 
        
        Args:
            path (str): Ruta desde donde cargar el modelo.
        """
        self._model = keras.models.load_model(path)
        
    def load_weights(self,path ):
        """
        Cargar pesos de un modelo. 
        
        Args:
            path (str): Ruta desde donde cargar los pesos del modelo.
        """
        self._model.load_weights(path)

        
    def predict(self, X):
        """
        Predecir usando un modelo.
        
        Args:
            X (array): Conjunto de datos del que realizar la predicción.
        
        Returns:
            array: Array de predicciones. 
        """
        return self._model.predict(X)
    
