


import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from mgmtfm.tokens_commons import numwords, commons, names
from unidecode import unidecode

class Clean:
  """
  Clase para el preprocesamiento de documentos que se destinaran al PLN.
  """    
  tokens = None 
  _combined = None
  distribucion_tipos = None
  X_train = None
  X_test = None
  y_train = None
  y_test = None
  tipos = None
  NUM_WORDS=20000
  word_index = None
  _class_binary = None
  label_dict = None


  def __init__(self):
    pass



  def balancing(self, x=None, class_binary=None, balancing_types=None):
        """
        Balancear los datos para que todas las clases que se usen en el entrenamiento tengan el mismo número de elementos. Y realizar un shuffle entre ellos.
    
        Args:
            x (DataFrame): Dataframe que se va a proceder a balancear. Se balanceará en función 
                del campo "tipo". Si fuera None usará los tokens actuales.
            class_binary (str): Establecer si estamos en un caso de clasificación binaria y queremos 
                balancear entre esa clase y el resto. 
            balancing_types (list<str>): Realiza lo mismo que en el caso anterior, pero con una 
                lista de clases realizando el balanceado entre esta lista y el resto. 
        
        Returns:
            DataFrame: Dataframe balanceado.
    
        """
        if (x):
            balanced = x
        elif (isinstance(self.tokens, pd.DataFrame)):
            balanced = self.tokens
            x = self.tokens
        else:
            print("No hay datos para balancear")
            return
        if (class_binary):
            x_class = x[x["tipo"] == class_binary]
            x_noclass = x[x["tipo"] != class_binary]
            len_max = len(x_class) if (len(x_class) < len(x_noclass)) else len(x_noclass)
            reduce_df = x_noclass if (len(x_class) < len(x_noclass)) else x_class
            nonreduce_df = x_class if (len(x_class) < len(x_noclass)) else x_noclass
            reduce_df =  reduce_df.loc[np.random.choice(reduce_df.index, len_max, replace=False)]
            balanced =reduce_df.append(nonreduce_df)
        else:
            tipos = np.unique(x["tipo"])
            if (balancing_types):
                tipos = balancing_types
            x_class_lists = [x[x["tipo"] == tipo] for tipo in tipos]
            if (balancing_types):
                x_class_lists.append(x[~x.tipo.isin(balancing_types)])
            min_class_len = min([len(x) for x in x_class_lists])
            x_class_lists = [x.loc[np.random.choice(x.index, min_class_len, replace=False)] for x in x_class_lists]
            balanced = pd.concat(x_class_lists)
            balanced = balanced.sample(frac=1).reset_index(drop=True)
            
        balanced = balanced.sample(frac=1).reset_index(drop=True)
        self._combined = balanced
        self.distribucion_tipos = (self._combined.groupby(['tipo']).count().reset_index()[['tipo','subtipo']]).set_index("tipo") 
        self.distribucion_tipos.columns = [ "count"]
        #si ya se han calculado los tokens
        if (isinstance(self.tokens, pd.DataFrame)):
            self.tokens=balanced         
        
        return balanced


     
  def quit_class(self, delete):
    """
    Eliminar una clase de los tokens actuales. 
    
    Args:
        delete (str): Clase que se eliminará.
    """
    #tiene que tener valor el token
    if (isinstance(self.tokens, pd.DataFrame)):
        x = self.tokens
        self.tokens = x[x["tipo"] != delete]
        self._combined = self.tokens
        self.distribucion_tipos = (self._combined.groupby(['tipo']).count().reset_index()[['tipo','subtipo']]).set_index("tipo") 
        self.distribucion_tipos.columns = [ "count"]
        
        
        
        
        
        
        
    
  def _labelEncoder(self, Y):
    """
    Realizar label encoder de una etiqueta. 
    
    Args:
        Y (array): Array o lista de etiquetas. 
        
    Returns:
        array: Array con el encoder realizado. 
    """
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    return encoded_Y
    
    
    
  def get_set_data(self, test_size=0.33, seed=2019, class_binary=None,  sequence=True, 
                   shuffle=True, max_seq_len=None, balancing_types=None):
    
    """
    Obtener conjuntos de datos de test y de entrenamiento, realizar el etiquetado de los conjuntos en 
        el caso que sea binario o haya balanceo. Realizar el padding si tratamos con secuencias. 
    
    Args:
        test_size (float): Porcentaje de datos de entrenamiento. Puede ser 0 para obtener los datos en un solo conjunto 
            y en el mismo orden (sin *shuffle*).
        seed (int): Semilla para reproducir los resultados aleatorios.
        class_binary (str): Válido para el etiquetado. Asigna un 1 a la clase binaria y un 0 al resto. 
        sequence (bool): Si se esta trabajando con secuencias o no. Se realiza  padding en el caso de que se active. *True* por defecto.
        shuffle (bool): Coger los datos en orden aleatorio(True) o secuencial (False). No afecta con test_size=0.
        max_seq_len (int): En el caso de usar secuencias la longitud máxima para el *padding*.
        balancing_types (list): Tipos que se van a usar para clasificar. El resto de tipos se etiquetaran como "Resto".
    """
    

    texts=self.tokens["seqtext"].values.tolist()

    self.tipos=list(self.distribucion_tipos.index.tolist())
    ## Los tipos de balanceo los pasamos al final para convertirlos en el resto
    rest_index = len(self.tipos)
    if balancing_types:
        rest_types = list(filter(lambda x: x not in balancing_types,self.tipos))
        self.tipos = balancing_types + rest_types
        rest_index = len(balancing_types)
    
    
    self.label_dict={}
    
    if class_binary:
        self._class_binary = class_binary
        for _,tipo in enumerate(self.tipos):
            self.label_dict[tipo]= (1 if (class_binary == tipo) else 0)
    else:
        for i,tipo in enumerate(self.tipos):
            self.label_dict[tipo]=i if i < rest_index else rest_index
        
        
    labels=self.tokens.tipo.apply(lambda x:self.label_dict[x])

    
    
    if test_size == 0:
        texts_train = texts
        labels_train = labels
        self.y_test = None
    else:
        texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=test_size, 
                                                                          random_state=seed, shuffle=shuffle)
        self.y_test =self._labelEncoder(labels_test) 
        
    self.y_train = self._labelEncoder(labels_train)
     
    
    if not sequence:
        self.X_train = texts_train
        if test_size == 0:
            self.X_test = None
        else: 
            self.X_test = texts_test
    else:
        if (not max_seq_len):
            max_seq_len = int(max(self.tokens["plaintext"].apply(len)))
        
            
        self.X_train = pad_sequences(texts_train, maxlen=max_seq_len)
        
        if test_size == 0:
            self.X_test = None
            print('Shape of X train:', self.X_train.shape)
            print('Shape of label train:', self.y_train.shape)
        else:
            self.X_test = pad_sequences(texts_test, maxlen=max_seq_len)
            print('Shape of X train and X validation tensor:', self.X_train.shape,  self.X_test.shape)
            print('Shape of label train and validation tensor:', self.y_train.shape, self.y_test.shape)
    
        
          
    return (self.X_train,self.y_train, self.X_test, self.y_test )



  def pad_Xs(self, maxlen):
        """
        Realizar el padding de los conjuntos de entrenamiento y test. 
        
        Args:
            maxlen (int): Longitud máxima para el padding.
            
        Returns:
            tuple (<Dataframe, Dataframe>): conjunto de entrenamiento y de test tras el padding.
        """
        X_train = pad_sequences(self.X_train, maxlen=maxlen)
        X_test = pad_sequences(self.X_test, maxlen=maxlen)
        return (X_train, X_test)
        


  def _tokenizer(self, x, quit_commons=True):
      """
      Aplicar el tokenizado a una cadena de texto. Pasa a minúsculas, elimina caracteres especiales, 
          stopwords, números, nombres propios. 
        
      Args:
          x (str): Cadena que tokenizar.
          quit_common (bool): Si se desean eliminar también una lista de palabras comunes. Por defecto: True.
    
      Returns: 
        list: Lista de tokens. 
    
      """
      toktok = ToktokTokenizer()
      common_words =[]
      if quit_commons:
          common_words = commons
      x_lower = x.lower().replace("o dos", "o2")
      tokens_not_filter = [unidecode(item.lower()) for item in toktok.tokenize(x)]
      tokens = [item for item in tokens_not_filter 
                if item not in stopwords.words('spanish')
                and item not in numwords
                and item not in common_words
                and item not in names
                and len(item)>2]
      return tokens
  


  
  # Limit, limita los eventos por fecha y coge los últimos
  def tokenize(self, quit_commons=True, limit=None):
    """
    Tokenizar el "plaintext" del *dataframe* "combined" utilizando el método privado *_tokenize*. 
      
    Args:
        quit_commons (bool): Eliminar palabras comunes al tokenizar.
        limit (int): Si se establece tokeniza unicamente este número de llamadas.
           
    Returns: 
        dataframe: Dataframe con los tokens. Tambien se almacenan en self.tokens.
       
    """
        
    combined_last = self._combined
    if limit:
      combined_last = self._combined.sort_values(by=['fx_evento'], ascending=False).head(limit)
    self.tokens = combined_last[["co_llamada_verint", "plaintext", "tipo", "subtipo"]]

    self.tokens['plaintext'] = self.tokens['plaintext'].apply(lambda x: self._tokenizer(x, quit_commons))
    
    tokenizer = Tokenizer(num_words=self.NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
    texts = self.tokens['plaintext'].values.tolist()
    tokenizer.fit_on_texts(texts)
    self.tokens['seqtext'] = tokenizer.texts_to_sequences(texts)
    self.word_index = tokenizer.word_index
    print('Found {:,} unique tokens.'.format(len(self.word_index)))
    return self.tokens



  def save_tokens(self, file_name):
    """
    Guardar los tokens en un fichero (pickle).
       
    Args:
        file_name (str): nombre del fichero en el que se almacenaran los tokens. 
    """
    self.tokens.to_pickle(file_name)
    
  def load_tokens(self, file_name):
    """
    Cargar los tokens desde un fichero. Además carga las variables "combined", 
        "distribucion_tipos" y "word_index".
    """
    self.tokens = pd.read_pickle(file_name)
    tokenizer = Tokenizer(num_words=self.NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
    texts = self.tokens['plaintext'].values.tolist()
    tokenizer.fit_on_texts(texts)
    self.tokens['seqtext'] = tokenizer.texts_to_sequences(texts)
    self._combined =self.tokens
    self.distribucion_tipos = (self._combined.groupby(['tipo']).count().reset_index()[['tipo','subtipo']]).set_index("tipo") 
    self.distribucion_tipos.columns = [ "count"]
    self.word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(self.word_index))
  
  def _get_ivr_combined(self,verint, ivr_hierarchy):
    """
    Combina el dataframe de verint con las jerarquías de IVR. 
    
        Args:
            verint (dataframe): Dataframe de Verint con las transcripciones de las llamadas.
            ivr_hierarchy (dataframe): Dataframe con las jerarquías de IVR (coge la última versión de la tabla)
        Returns: 
            dataframe: Join de ambos dataframes.
    
    """
    last_date = ivr_hierarchy["fx_carga"].max()
    print("Nos quedamos con la fecha: {}".format(last_date))
    ivr_hierarchy = ivr_hierarchy[ivr_hierarchy["fx_carga"] == last_date].drop(columns="fx_carga")
    ivr_hierarchy.columns = ["tipo", "subtipo", "no_destino_pa"]
    combined = pd.merge(verint, ivr_hierarchy, on=['no_destino_pa'], how='inner')
    return combined
  
  
  
  def _get_monitor_combined(self,verint, monitorizaciones):
    """
    Combina el dataframe de verint con los datos de monitorización. 
    
      Args:
          verint (dataframe): Dataframe de Verint con las transcripciones de las llamadas.
          monitorizaciones (dataframe): Dataframe con los datos de monitorización.
      Returns: 
          dataframe: Join de ambos dataframes.
    
    """
    monitor = monitorizaciones[(monitorizaciones.name.str.startswith('C') | monitorizaciones.name.str.startswith('D'))].drop_duplicates()
    diccionario = {'C#1':'información', 'C#2':'contratar', 'D#1':'información', 'D#2':'consulta', 'D#3':'queja', 'D#4':'trámite'}
    for k in diccionario.keys():
      monitor.loc[monitor.name.str.startswith(k), 'tipo'] = diccionario[k]
          
    monitor.name = monitor.name.apply(lambda x:  'comercial' if x.startswith('C') else 'no_comercial')
    monitor.dropna(inplace=True)
    monitor.it_llamada = monitor.it_llamada.dt.date
    monitor.columns = ['co_llamada_verint', 'ucid', 'fx_evento', 'duration', 'unidad_negocio', 'motivo_llamada', 'subtipo', 'tipo']
    combined = pd.merge(verint, monitor, on=['co_llamada_verint', 'ucid', 'fx_evento', 'duration'], how='inner')
    return combined
  
  
  def get_combined_and_distribution(self,calls, types,ivr=False):
    """
    Combinar el dataframe de llamadas con el dataframe de tipos correspondiente. 
    
      Args:
          verint (dataframe): Dataframe de Verint con las transcripciones de las llamadas.
          tipos (dataframe): Dataframe con los tipos.
          ivr (boolean): Si es True el dataframe de tipos será el de IVR, de no ser así será el de monitorizaciones.
      Returns: 
          dataframe: Join de ambos dataframes.
    
    """
        
    if ivr:
        self._combined =  self._get_ivr_combined(calls, types)
    else:
        self._combined =  self._get_monitor_combined(calls, types)
  
    self.distribucion_tipos = (self._combined.groupby(['tipo']).count().reset_index()[['tipo','subtipo']]).set_index("tipo") #.size().reset_index().groupby('col2')[[0]].max()
    self.distribucion_tipos.columns = [ "count"]
  

    return (self._combined, self.distribucion_tipos)
  
  
  
  def _one_hot(self, y):
    """
    Realizar la conversión de una lista de etiquetas al formato one_hot.
    
    Args:
        y (array):  Array de etiquetas.
    
    Returns:
        array: Array en formato one_hot. 
    
    
    """
    one_vec = np.zeros((y.size,np.unique(y).size))
    one_vec[np.arange(y.size), y] = 1
    return one_vec
  
  
  def one_hot_ys(self): 
    """
    Realizar la conversión de las etiquetas de entrenamiento y test al formato one_hot.
    
    
    Returns:
        (array, array): Tupla de arrays en formato one_hot. 
    
    
    """
    y_train = self._one_hot(self.y_train)
    y_test = self._one_hot(self.y_test)
    return (y_train, y_test)
  
  
  
  
  
  
  
  
  