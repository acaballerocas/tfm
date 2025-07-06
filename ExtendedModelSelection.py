'''
Importar librerías necesarias para las clases.
Nota: usar Pipeline de imblearn, no de sklearn.
'''

from typing import Any, Dict, List, Tuple, Optional, Union
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# EXTENDED GRID SEARCH #

class ExtendedGridSearchCV:
    def __init__(self, 
                 estimator: BaseEstimator, 
                 param_grid: Dict[str, Any], 
                 samplers: List[Tuple[str, Any]], 
                 sampler_params: Dict[str, Any], 
                 **kwargs: Any
                 ) -> None:
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
        - cv_results_df_: objeto para guardar los resultados completos de la búsqueda como DataFrame
        '''
        self.estimator = estimator
        self.param_grid = param_grid
        self.samplers = samplers
        self.sampler_params = sampler_params
        self.kwargs = kwargs
        self.cv_results_df_ = Optional[pd.DataFrame] = None

    def fit(self, 
            X: Union[pd.DataFrame, Any], 
            y: Union[pd.Series, Any]
            ) -> Tuple[BaseEstimator, Dict[str, Any], float]:
        best_score = -float('inf')  #Inicializamos con valor infinito negativo
        best_estimator = None
        best_params = None
        all_dfs = []


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

            # Guardamos el resultado de CV como DataFrame
            df = pd.DataFrame(ext_grid_search.cv_results_)
            df['sampler'] = samplername
            all_dfs.append(df)

            
            # Actualizar el mejor resultado si es necesario
            if ext_grid_search.best_score_ > best_score:
                best_score = ext_grid_search.best_score_
                best_estimator = ext_grid_search.best_estimator_
                best_params = ext_grid_search.best_params_
  
        # Unir resultados en un único data frame
        self.cv_results_df_ = pd.concat(all_dfs, ignore_index=True)
        # Modificamos el ranking para que considere simultáneamente todas las búsquedas realizadas
        self.cv_results_df_['rank_test_score'] = self.cv_results_df_['mean_test_score'].rank(ascending=False, method='dense').astype(int)


        return best_estimator, best_params, best_score
    
    # Obtención de los resultados completos en data frame
    def get_cv_results_df(self) -> Optional[pd.DataFrame]:
      return self.cv_results_df_



# EXTENDED RANDOMIZED SEARCH #

class ExtendedRandomizedSearchCV:
    def __init__(self, 
                 estimator: BaseEstimator, 
                 param_grid: Dict[str, Any], 
                 samplers: List[Tuple[str, Any]], 
                 sampler_params: Dict[str, Any], 
                 **kwargs: Any
                 ) -> None:
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
        - cv_results_df_: objeto para guardar los resultados completos de la búsqueda como DataFrame
        '''
        self.estimator = estimator
        self.param_grid = param_grid
        self.samplers = samplers
        self.sampler_params = sampler_params
        self.kwargs = kwargs
        self.cv_results_df_: Optional[pd.DataFrame] = None
        
    def fit(self, 
            X: Union[pd.DataFrame, Any], 
            y: Union[pd.Series, Any]
            ) -> Tuple[BaseEstimator, Dict[str, Any], float]:
        best_score = -float('inf')  #Mejor score. Inicializamos con valor infinito negativo.
        best_estimator = None #Mejor combinación de Estrategia+Modelo. Iniciar en None.
        best_params = None #Mejores hiperparámetros. Iniciar en None.
        all_dfs = []


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

            # Guardamos el resultado de CV como DataFrame
            df = pd.DataFrame(ext_random_search.cv_results_)
            df['sampler'] = samplername
            all_dfs.append(df)


            # Actualizar el mejor resultado si es necesario
            if ext_random_search.best_score_ > best_score:
                best_score = ext_random_search.best_score_
                best_estimator = ext_random_search.best_estimator_
                best_params = ext_random_search.best_params_

        # Unir resultados en un único data frame
        self.cv_results_df_ = pd.concat(all_dfs, ignore_index=True)
        # Modificamos el ranking para que considere simultáneamente todas las búsquedas realizadas
        self.cv_results_df_['rank_test_score'] = self.cv_results_df_['mean_test_score'].rank(ascending=False, method='dense').astype(int)

        return best_estimator, best_params, best_score
    
    # Obtención de los resultados completos en data frame
    def get_cv_results_df(self) -> Optional[pd.DataFrame]:
      return self.cv_results_df_
