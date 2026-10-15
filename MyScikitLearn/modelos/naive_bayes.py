import numpy as np
from ..clasificador import Clasificador

class ClasificadorNaiveBayes(Clasificador):
    def __init__(self, laplace=True):
        self.laplace = laplace
        self.apriori = {}
        self.probabilidades_discretas = {}
        self.estadisticas_continuas = {}
        self.clases_posibles = []

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        clases_reales = datosTrain[:, -1]
        self.clases_posibles = np.unique(clases_reales)
        total_filas = len(datosTrain)
        num_atributos = datosTrain.shape[1] - 1
        
        # 1. Probabilidades a priori
        for clase in self.clases_posibles:
            self.apriori[clase] = np.sum(clases_reales == clase) / total_filas
            3
        # 2. Entrenamiento por atributos
        for clase in self.clases_posibles:
            datos_clase = datosTrain[clases_reales == clase]
            
            for col in range(num_atributos):
                columna_datos = datos_clase[:, col]
                
                if atributosDiscretos[col]:
                    valores_totales_col = np.unique(datosTrain[:, col])
                    stats = self._entrenar_discreto(columna_datos, valores_totales_col)
                    
                    if col not in self.probabilidades_discretas:
                        self.probabilidades_discretas[col] = {}
                    self.probabilidades_discretas[col][clase] = stats
                else:
                    stats = self._entrenar_continuo(columna_datos)
                    
                    if col not in self.estadisticas_continuas:
                        self.estadisticas_continuas[col] = {}
                    self.estadisticas_continuas[col][clase] = stats

    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        predicciones = []
        num_atributos = datosTest.shape[1] - 1
        
        for fila in datosTest:
            probabilidades_fila = {}
            
            for clase in self.clases_posibles:
                score_clase = np.log(self.apriori[clase])
                
                for col in range(num_atributos):
                    valor = fila[col]
                    
                    if atributosDiscretos[col]:
                        diccionario_probs = self.probabilidades_discretas[col][clase]
                        score_clase += self._log_prob_discreta(valor, diccionario_probs)
                    else:
                        stats_continuas = self.estadisticas_continuas[col][clase]
                        score_clase += self._log_prob_continua(valor, stats_continuas)
                
                probabilidades_fila[clase] = score_clase
                
            clase_predicha = max(probabilidades_fila, key=probabilidades_fila.get)
            predicciones.append(clase_predicha)
            
        return np.array(predicciones)

    # --- MÉTODOS PRIVADOS ---

    def _entrenar_discreto(self, columna_datos_clase, valores_posibles_globales):
        probabilidades = {}
        total_clase = len(columna_datos_clase)
        num_valores_posibles = len(valores_posibles_globales)
        
        for valor in valores_posibles_globales:
            apariciones = np.sum(columna_datos_clase == valor)
            if self.laplace:
                prob = (apariciones + 1) / (total_clase + num_valores_posibles)
            else:
                prob = apariciones / total_clase
            probabilidades[valor] = prob
            
        return probabilidades

    def _entrenar_continuo(self, columna_datos_clase):
        return {
            'media': np.mean(columna_datos_clase),
            'varianza': np.var(columna_datos_clase)
        }

    def _log_prob_discreta(self, valor, diccionario_probs):
        prob = diccionario_probs.get(valor, 1e-10)
        if prob == 0: 
            prob = 1e-10
        return np.log(prob)

    def _log_prob_continua(self, valor, stats):
        media = stats['media']
        varianza = stats['varianza']
        
        if varianza == 0: 
            varianza = 1e-10
            
        exponente = np.exp(-((valor - media) ** 2) / (2 * varianza))
        prob = (1 / np.sqrt(2 * np.pi * varianza)) * exponente
        
        if prob == 0: 
            prob = 1e-10
        return np.log(prob)