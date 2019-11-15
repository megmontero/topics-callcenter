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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument




class Embedding():
    """
    Clase encargada de realizar y gestionar los embedding. Actualmente existen 
    las opciones de Word2Vec (CBOW y Skip-Gram) o Doc2VEC.
    """
    vocabulary_size = 0
    _word_vectors = None
    embedding_dim = 100
    _NUM_WORDS=20000
    _word_index = None
    _train_data = None 
    _train_from_verint=True
    _embedding_matrix = None
    _doc2vec_model = None
    _type=1
    
    def __init__(self, train_data, word_index,num_words=20000, train_from_verint=True):
        """
        Constructor de la clase. 
        
        Args:
            train_data (dataframe): Dataframe que se usará para entrenar el embedding.
            word_index (dict): Diccionario de palabras del vocabulario.
            num_words (int): Número máximo e palabras. 
            train_from_verint (bool):  Si se va a entrenar los embedding desde el dataset de Verint.
        """
        self._NUM_WORDS = num_words
        self._word_index=word_index
        if (train_from_verint):
            self._train_data = train_data["plaintext"].values.tolist()
        
        
        
        
    def train_embedding(self, min_count=1, size=100, workers=16, window=5, type=1 ):
        """
        Entrenar el embedding para word2vec o doc2vec.
        
        Args:
            min_count (int): Número mínimo de veces que debe aparecer una palabra para ser tenida en cuenta.
            size (int): Dimension del vector de embedding.
            workers (int): Número de workers usados en el entrenamiento.
            window (int):  Tamaño de la ventana de palabras para el entrenamiento. 
            type(int): 0-> CBOW 1-> SKIP-GRAM (default) 2-> Doc2Vec
        """
        self._type = type
        
        if type == 0 or type==1: 
            self._word_vectors = Word2Vec(self._train_data,min_count=min_count,size=size, workers=workers,window=window,sg=type)
            self.vocabulary_size = len(self._word_vectors.wv.vocab) +1
        elif type==2:
            documents =  [TaggedDocument(doc, [i]) for i, doc in enumerate(self._train_data)]
            self._doc2vec_model = Doc2Vec(documents, vector_size=size, window=window, 
                                          min_count=min_count, workers=workers)
            
        self.embedding_dim = size
        
    
    def save_embedding(self, path):
        """
        Almacenar el embedding (tras el entrenamiento).
        
        Args:
            path (str): Ruta del fichero. 
        """
        if (self._type==2):
            self._doc2vec_model.save(path)
        elif(self._type ==0 or self._type==1):
            self._word_vectors.save(path)
            
    def doc2vec_infer(self, doc):
        """
        Inferir el vector Doc2Vec de un documento. 
        
        Args: 
            doc (str): Documento de texto a inferir.
        
        Returns:
            Array: Vector Doc2Vec.
        """
        if (self._type==2):
            return self._doc2vec_model.infer_vector(doc)
        return None
    
            
    def get_embedding_matrix(self):
        """
        Obtener la matriz de embeddings. 
        
        Returns: 
            array<array>: Matriz de embeddings. 
            
        """
        if (self._embedding_matrix):
            return self._embedding_matrix
        
        self._embedding_matrix = np.zeros((self.vocabulary_size, self.embedding_dim))
        for word, i in self._word_index.items():
            if i>=self._NUM_WORDS:
                continue
            try:
                embedding_vector = self._word_vectors[word]
                self._embedding_matrix[i] = embedding_vector
            except KeyError:
                self._embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),self.embedding_dim)
        return self._embedding_matrix
    
    def load_embedding(self, path, type=1):
        """
        Cargar embedding desde un fichero. 
        
        Args:
            path (str): Ruta de la que cargar el embedding. 
            type (int): 0-> CBOW 1-> SKIP-GRAM (default) 2-> Doc2Vec
        """
        
        self._type=type
        if (self._type==2):
            self._doc2vec_model = Doc2Vec.load(path)
        elif(self._type ==0 or self._type==1):
            self._word_vectors = Word2Vec.load(path)
            self.vocabulary_size = len(self._word_vectors.wv.vocab) +1
            self.embedding_dim = self._word_vectors.vector_size
        
        