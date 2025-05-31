'''
Importar librerías necesarias para las clases.
Nota: usar Pipeline de imblearn, no de sklearn.
'''

from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# EXTENDED GRID SEARCH #

class ExtendedGridSearchCV:
    def __init__(self, estimator, param_grid, samplers, sampler_params, **kwargs):
        '''
        Definiciones:
        - estimator: algoritmo de aprendizaje automático de sklearn
        - param_grid: diccionario de hiperparámetros del algoritmo/modelo.
        - samplers: lista de tuplas con todas las estrategias consideradas.
          Formato tupla: (nombre del sampler, función de sobre/submuestreo de imblearn)
          (el nombre es importante para el diccionario sampler_params)
        - sampler_params: diccionario de parámetros del sampler.
          Las keys deben tener la forma nombresampler__nombreparametro.
        - **kwargs: argumentos adicionales de GridSearch
        '''
        self.estimator = estimator
        self.param_grid = param_grid
        self.samplers = samplers
        self.sampler_params = sampler_params
        self.kwargs = kwargs

    def fit(self, X, y):
        best_score = -float('inf')  #Inicializamos con valor infinito negativo
        best_estimator = None
        best_params = None


        # Para cada función de sobremuestreo/submuestreo, es necesario crear un pipeline
        # y ejecutar un GridSearch independiente. 
        # Posteriormente se comparan "manualmente" los resultados.
        for samplername, sampler in self.samplers:
            # Crear pipeline (usando Pipeline de imblearn)
            pipeline = Pipeline([(samplername, sampler), ('model', self.estimator)])

            # Agregar todos los hiperparámetros (estrategia+modelo) en un mismo diccionario
            # Nota: no todos los parámetros de sampler_params tienen por qué ser compatibles
            # con todas las estrategias. Seleccionar solo los que sí lo son.
            ext_param_grid = {}
            # Añadir hiperparámetros del modelo
            for key, value in self.param_grid.items():
                    ext_param_grid[f'model__{key}'] = value
            # Añadir hiperparámetros de la estrategia (solo los que corresponda)
            for key, value in self.sampler_params.items():
                    if key.startswith(f'{samplername}__'):
                        ext_param_grid[key] = value

            # Inicializar GridSearchCV con Pipeline de Imblearn
            ext_grid_search = GridSearchCV(pipeline, ext_param_grid, **self.kwargs)
            ext_grid_search.fit(X, y)

            # Actualizar el mejor resultado si es necesario
            if ext_grid_search.best_score_ > best_score:
                best_score = ext_grid_search.best_score_
                best_estimator = ext_grid_search.best_estimator_
                best_params = ext_grid_search.best_params_

        return best_estimator, best_params, best_score


# EXTENDED RANDOMIZED SEARCH #

class ExtendedRandomizedSearchCV:
    def __init__(self, estimator, param_grid, samplers, sampler_params, **kwargs):
        '''
        Definiciones:
        - estimator: algoritmo de aprendizaje automático de sklearn
        - param_grid: diccionario de hiperparámetros del algoritmo/modelo
        - samplers: lista de tuplas con todas las estrategias consideradas.
          Formato tupla: (nombre del sampler, función de sobre/submuestreo de imblearn)
          (el nombre es importante para el diccionario sampler_params)
        - sampler_params: diccionario de parámetros del sampler.
          Las keys deben tener la forma nombresampler__nombreparametro.
        - **kwargs: argumentos adicionales de GridSearch
        '''
        self.estimator = estimator
        self.param_grid = param_grid
        self.samplers = samplers
        self.sampler_params = sampler_params
        self.kwargs = kwargs
        
    def fit(self, X, y):
        best_score = -float('inf')  #Mejor score. Inicializamos con valor infinito negativo.
        best_estimator = None #Mejor combinación de Estrategia+Modelo. Iniciar en None.
        best_params = None #Mejores hiperparámetros. Iniciar en None.


        # Para cada función de sobremuestreo/submuestreo, es necesario crear un pipeline
        # y ejecutar un GridSearch independiente. 
        # Posteriormente se comparan "manualmente" los resultados.
        for samplername, sampler in self.samplers:
            # Crear pipeline (usando Pipeline de imblearn)
            pipeline = Pipeline([(samplername, sampler), ('model', self.estimator)])

            # Agregar todos los hiperparámetros (estrategia+modelo) en un mismo diccionario
            # Nota: no todos los parámetros de sampler_params tienen por qué ser compatibles
            # con todas las estrategias. Seleccionar solo los que sí lo son.
            ext_param_grid = {}
            # Añadir hiperparámetros del modelo
            for key, value in self.param_grid.items():
                    ext_param_grid[f'model__{key}'] = value
            # Añadir hiperparámetros de la estrategia (solo los que corresponda)
            for key, value in self.sampler_params.items():
                    if key.startswith(f'{samplername}__'):
                        ext_param_grid[key] = value

            # Inicializar GridSearchCV con Pipeline de Imblearn
            ext_random_search = RandomizedSearchCV(pipeline, ext_param_grid, **self.kwargs)
            ext_random_search.fit(X, y)

            # Actualizar el mejor resultado si es necesario
            if ext_random_search.best_score_ > best_score:
                best_score = ext_random_search.best_score_
                best_estimator = ext_random_search.best_estimator_
                best_params = ext_random_search.best_params_

        return best_estimator, best_params, best_score





