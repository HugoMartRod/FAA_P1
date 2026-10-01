import numpy as np
from scipy.stats import norm 

class Clasificador:
    """
    Clase base para todos los algoritmos de Machine Learning. 
    Define el ciclo de vida y las operaciones obligatorias.
    """
    
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        """
        Construye el modelo a partir de los datos de entrenamiento.
        Debe ser sobreescrito por las clases hijas.
        """
        raise NotImplementedError("El método de entrenamiento debe ser implementado por el algoritmo específico.")
    
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        """
        Evalúa los datos de prueba y devuelve las predicciones.
        Debe ser sobreescrito por las clases hijas.
        """
        raise NotImplementedError("El método de clasificación debe ser implementado por el algoritmo específico.")
    
    def error(self, datos, predicciones):
        """
        Calcula el porcentaje de fallos comparando las clases reales 
        con las predicciones devueltas por el modelo.
        Común a todos los clasificadores.
        """
        # Asumimos que la clase real es la última columna de los datos
        clases_reales = datos[:, -1]
        
        # Contamos cuántos fallos hay comparando los arrays
        fallos = sum(clases_reales != predicciones)
        
        # Devolvemos el porcentaje de error
        porcentaje_error = fallos / len(clases_reales)
        return porcentaje_error

class ClasificadorNaiveBayes(Clasificador):
    def __init__(self, laplace=True):
        """
        Inicializa el clasificador indicando si se usa la corrección de Laplace.
        """
        self.laplace = laplace
        
        # Diccionarios para almacenar la "memoria" del modelo
        self.apriori = {}                  # {clase: probabilidad_apriori}
        self.probabilidades_discretas = {} # {columna: {clase: {valor_atributo: probabilidad}}}
        self.estadisticas_continuas = {}   # {columna: {clase: {'media': media, 'varianza': varianza}}}
        self.clases_posibles = []          # Lista de clases únicas en el dataset

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        """
        Construye el modelo calculando probabilidades a priori, frecuencias 
        para atributos discretos y medias/varianzas para continuos.
        """
        # La clase real siempre está en la última columna
        clases_reales = datosTrain[:, -1]
        self.clases_posibles = np.unique(clases_reales)
        total_filas = len(datosTrain)
        
        # 1. CÁLCULO DE PROBABILIDADES A PRIORI
        for clase in self.clases_posibles:
            # Sumamos el total de veces que aparece cada clase
            num_apariciones = np.sum(clases_reales == clase)
            # Calculamos la probabilidad a priori y la añadimos al diccionario 
            self.apriori[clase] = num_apariciones / total_filas

        
        # 2. CÁLCULO DE ESTADÍSTICAS POR ATRIBUTO
        num_atributos = datosTrain.shape[1] - 1 # Sin contar la última columna (clase)
        
        for clase in self.clases_posibles:
            # Filtramos solo las filas que pertenecen a esta clase
            datos_clase = datosTrain[clases_reales == clase]
            
            for col in range(num_atributos):
                if atributosDiscretos[col]:
                    # ES DISCRETO: Calcular frecuencias con o sin Laplace

                    # Extraemos la columna actual solo para los clientes de esta clase
                    columna_datos = datos_clase[:, col]
                    
                    # Vemos cuántos valores únicos tiene este atributo en TODO el dataset de entrenamiento (|V|)
                    valores_posibles = np.unique(datosTrain[:, col])
                    num_valores_posibles = len(valores_posibles)
                    total_clase = len(columna_datos)
                    
                    # Preparamos el "archivador" para esta columna y clase
                    if col not in self.probabilidades_discretas:
                        self.probabilidades_discretas[col] = {}
                    self.probabilidades_discretas[col][clase] = {}
                    
                    # Calculamos la probabilidad para CADA valor posible
                    for valor in valores_posibles:
                        # Contamos cuántas veces aparece exactamente este valor
                        apariciones = np.sum(columna_datos == valor)
                        
                        if self.laplace:
                            # Fórmula CON Corrección de Laplace
                            prob = (apariciones + 1) / (total_clase + num_valores_posibles)
                        else:
                            # Fórmula SIN Corrección de Laplace
                            prob = apariciones / total_clase
                            
                        # Guardamos el resultado en nuestro diccionario
                        self.probabilidades_discretas[col][clase][valor] = prob
                else:
                    # ES CONTINUO: Calcular media y varianza

                    # Extraemos únicamente la columna actual de los datos ya filtrados por clase
                    columna_datos = datos_clase[:, col]
                    
                    # Calculamos la estadística de la distribución normal
                    media = np.mean(columna_datos)
                    varianza = np.var(columna_datos)
                    
                    # Creamos la clave de la columna en el diccionario si aún no existe
                    if col not in self.estadisticas_continuas:
                        self.estadisticas_continuas[col] = {}
                        
                    # Guardamos la media y varianza asociadas a esta clase
                    self.estadisticas_continuas[col][clase] = {'media': media, 'varianza': varianza}

    def clasifica(self, datosTest, atributosDiscretos, diccionario):
            """
            Evalúa los datos de prueba devolviendo una lista de predicciones.
            """
            predicciones = []
            num_atributos = datosTest.shape[1] - 1 # Ignoramos la última columna (la clase real)
            
            for fila in datosTest:
                # Aquí guardaremos el score (logaritmo de la probabilidad) para cada clase
                probabilidades_fila = {}
                
                for clase in self.clases_posibles:
                    # 1. Partimos del logaritmo de la probabilidad a priori
                    score_clase = np.log(self.apriori[clase])
                    
                    # 2. Sumamos los logaritmos de los atributos de la fila actual
                    for col in range(num_atributos):
                        valor = fila[col]
                        
                        if atributosDiscretos[col]:
                            # ES DISCRETO: Buscamos la frecuencia en el diccionario
                            # Si el valor no existía en el entrenamiento, asignamos una probabilidad ínfima 
                            # (1e-10) en lugar de 0 para que np.log no lance error
                            prob = self.probabilidades_discretas[col][clase].get(valor, 1e-10)
                            
                            # Si estamos ejecutando SIN Laplace y la prob es 0 puro, también la ajustamos
                            if prob == 0: 
                                prob = 1e-10
                                
                            score_clase += np.log(prob)
                            
                        else:
                            # ES CONTINUO: Evaluamos la función de densidad de la Normal "a mano"
                            media = self.estadisticas_continuas[col][clase]['media']
                            varianza = self.estadisticas_continuas[col][clase]['varianza']
                            
                            # Control de seguridad: evitar dividir entre cero si la varianza es 0
                            if varianza == 0: 
                                varianza = 1e-10
                            
                            # Fórmula matemática de la PDF de Gauss (¡Muchísimo más rápido que SciPy!)
                            exponente = np.exp(-((valor - media) ** 2) / (2 * varianza))
                            prob = (1 / np.sqrt(2 * np.pi * varianza)) * exponente
                            
                            if prob == 0: 
                                prob = 1e-10
                                
                            score_clase += np.log(prob)
                    
                    # Guardamos la puntuación total de esta clase
                    probabilidades_fila[clase] = score_clase
                    
                # 3. Seleccionamos la clase con la puntuación más alta
                # max() con key=dict.get devuelve la llave (clase) con el valor máximo
                clase_predicha = max(probabilidades_fila, key=probabilidades_fila.get)
                predicciones.append(clase_predicha)
                
            return np.array(predicciones)