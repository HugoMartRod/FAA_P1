import numpy as np
from ..clasificador import Clasificador # Ajusta la ruta a "from MyScikitLearn.clasificador import Clasificador" si no usas carpetas

class ClasificadorKNN(Clasificador):
    def __init__(self, k=5):
        """
        Inicializa el clasificador K-NN definiendo el número de vecinos (K).
        """
        self.k = k
        self.datos_entrenamiento = None
        self.clases_entrenamiento = None

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        """
        K-NN es un "lazy learner". El entrenamiento consiste únicamente en 
        almacenar los datos y sus clases reales para calcular distancias después.
        """
        # Guardamos los atributos (X) excluyendo la última columna
        self.datos_entrenamiento = datosTrain[:, :-1]
        # Guardamos las clases reales (y) que están en la última columna
        self.clases_entrenamiento = datosTrain[:, -1]

    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        """
        Evalúa los datos de prueba buscando los K vecinos más cercanos mediante 
        distancia Euclídea y asignando la clase mayoritaria.
        """
        predicciones = []
        # Extraemos los atributos de test ignorando la clase real
        X_test = datosTest[:, :-1]
        
        for fila_test in X_test:
            # 1. Calcular la distancia Euclídea de esta fila contra TODAS las filas de entrenamiento
            # Se usa vectorización de NumPy: np.sqrt(np.sum((Train - test)^2, axis=1))
            distancias = np.sqrt(np.sum((self.datos_entrenamiento - fila_test) ** 2, axis=1))
            
            # 2. Obtener los índices de los K vecinos con menor distancia
            # np.argsort devuelve las posiciones de los elementos ordenados de menor a mayor
            indices_vecinos = np.argsort(distancias)[:self.k]
            
            # 3. Consultar las clases reales a las que pertenecen esos K vecinos
            clases_vecinos = self.clases_entrenamiento[indices_vecinos]
            
            # 4. Votación por mayoría: np.unique devuelve las clases y su conteo
            clases_unicas, conteos = np.unique(clases_vecinos, return_counts=True)
            clase_ganadora = clases_unicas[np.argmax(conteos)]
            
            predicciones.append(clase_ganadora)
            
        return np.array(predicciones)