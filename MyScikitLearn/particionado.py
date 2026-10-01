import numpy as np

class EstrategiaParticionado:
    """
    Clase base que define la estructura obligatoria para cualquier 
    estrategia de validación futura.
    """
    def __init__(self):
        # Aquí guardaremos una lista de diccionarios: [{"train": [...], "test": [...]}]
        self.particiones = []

    def creaParticiones(self, numero_filas):
        # Este método lo sobreescribirán las clases hijas
        raise NotImplementedError("Este método debe ser implementado por las clases hijas")


class ValidacionSimple(EstrategiaParticionado):
    """
    Implementa la estrategia de validación simple (hold-out).
    """
    def __init__(self, proporcion_test=0.3, numero_ejecuciones=1):
        super().__init__() # Llama al constructor de la clase padre
        self.proporcion_test = proporcion_test
        self.numero_ejecuciones = numero_ejecuciones

    def creaParticiones(self, numero_filas):
        # Limpiamos particiones previas
        self.particiones = []
        
        # En la validación simple es común realizar varias ejecuciones para diferentes permutaciones
        for _ in range(self.numero_ejecuciones):
            # IMPORTANTE: Permutamos los datos para evitar sesgos en la distribución realista de las clases
            indices_aleatorios = np.random.permutation(numero_filas)
            
            # Calculamos el índice exacto donde hacer el corte
            corte = int(numero_filas * (1 - self.proporcion_test))
            
            # Separamos el array de índices permutados en dos bloques
            indices_train = indices_aleatorios[:corte].tolist()
            indices_test = indices_aleatorios[corte:].tolist()
            
            # Guardamos esta partición
            self.particiones.append({"train": indices_train, "test": indices_test})

class ValidacionCruzada(EstrategiaParticionado):
    """
    Implementa la estrategia de validación cruzada (K-Fold).
    """
    def __init__(self, numero_particiones=5):
        super().__init__()
        self.numero_particiones = numero_particiones

    def creaParticiones(self, numero_filas):
        # Limpiamos particiones previas
        self.particiones = []
        
        # 1. Barajamos los índices para evitar sesgos al igual que en hold-out
        indices_aleatorios = np.random.permutation(numero_filas)
        
        # 2. Dividimos el array de índices en K grupos (folds) lo más iguales posible
        folds = np.array_split(indices_aleatorios, self.numero_particiones)
        
        # 3. Construimos las K particiones
        for i in range(self.numero_particiones):
            # El fold actual es para test (lo convertimos a lista)
            indices_test = folds[i].tolist()
            
            # Los demás folds son para train (concatenamos y convertimos a lista)
            folds_train = [folds[j] for j in range(self.numero_particiones) if j != i]
            indices_train = np.concatenate(folds_train).tolist()
            
            # Guardamos la partición
            self.particiones.append({"train": indices_train, "test": indices_test})