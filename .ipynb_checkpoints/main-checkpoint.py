# -*- coding: utf-8 -*-
"""
Este es el main del esquema de un proyecto en el que se hace la predicción con una serie de modelos 
empleando aprendizaje supervisado, todos los procesos  de analytics deberían tener
una estructura similar. Más o menos las carpetas de este esquema y un main, que al ejecutar este fichero 
se dispararan todos los demás procesos y subprocesos. 

Deberemos emplear variables de entorno (alojadas en un archivo .env para modificar el comportamiento de nuestro 
proyecto sin tener que cambiar ni una sola linea de código
"""

import os
from dotenv import load_dotenv
load_dotenv('.env')

import pandas as pd
import numpy as np

from Carga_datos.main_carga_datos import carga_datos
from Estandarizacion_datos.main_estandarizacion_datos import estandarizacion_datos
#from Auditoria.main_auditoria import auditoria
from Predict.main_predict import predict
#from Delivery.main_delivery import delivery
from Train.main_train import train


#Emplearemos logs en vez de prints para los comentarios o mensajes de error, warnings, etc
import Utilidades.logs as lg 
log_name_main = lg.configure_FileLogger(name= __name__ , level=os.environ['LEVEL'] , filename=os.environ['PATH_LOGS_PRO'], stdout=os.environ['STDOUT_PRO']) #nombre de la variable que llevará los logs


""" NOTAS
    - Al log le debemos poner un nombre, en este caso le hemos llamado log_name_main, aunque habitualmente
    se usa un diminutivo del nombre del script para esto.
    Ejemplo: mn = lg.configure_FileLogger(name= __name__ , level=os.environ['LEVEL'] , filename=os.environ['PATH_LOGS_PRO'], stdout=os.environ['STDOUT_PRO']) #nombre de la variable que llevará los logs

    - Las salidas de cada uno de los procesos que se llaman desde el main deben ser única, habitualmente un diccionario
        
    - Importante! Generar los docstrings en las funciones para documentar cada una de ellas..
    Su objetivo... la descripción de sus inputs/outputs, y de la propia función
    
    - Las funciones deberán ir con su correcto control de errores y no debería haber 
    prints en el código, sino logs, que pueden tener diferentes niveles de importancia, 
    como se muestra a continuación: 
        
        lg.debug(dc1,'Este es un ejemplo de mensaje para debugear el código')
        lg.warning(dc1,'Este es un ejemplo de mensaje para dejar registrado un warning en la ejecución del código')
        lg.error(dc1,'Este es un ejemplo de mensaje para dejar registrado un error en la ejecución del código')
        lg.info(dc1,'Este es un ejemplo de mensaje para dejar registrada información importante de la ejecución')
        
    - Poner el mismo nombre a los parámetros de entrada de las funciones que a lo que se usa dentro de la función ayuda en el proceso de debugear
    por ejemplo: (OPCIONAL) 
    
        def hola(input1):
            return output1
        
        output1 = hola(input1)
"""


def main(): 
    """
    Se trata del main de esquema_inferir. Aquí se llama a los procesos que se han de
    ejecutarse.

    
    """
    lg.info(log_name_main,'Iniciando ejecución esquema train...')

    
    
    """
    Carga de Datos
    """
    carga_datos()
    
    """
    Estandarización de datos
    """
    estandarizacion_datos(input_folder=os.environ['INPUT_FOLDER_IMAGENES'],output_folder=os.environ['OUTPUT_FOLDER_IMAGENES'])


    """
    Auditoria
    """
    #dict_audit_inputs = auditoria(dict_inputs, conn)


    """
    Procesado de Datos
    """
    #processing(dict_inputs, conn)


    """
    Feature Engineering
    """
    #dict_inputs = feature_eng(dict_inputs, conn)

    """
    Entrenamiento
    """
    training = os.environ['TRAIN'].lower() == "true"
    if training:
        cnn = train(save_mdl=os.environ['SAVE_MDL'])
    
    """
    Predict
    """
    prediction_val = os.environ['PREDICTION'].lower() == "true"
    if prediction_val:
        if training:
            predict(cnn)
        else:
            predict()

    
    """
    Delivery
    """    
    #delivery(dict_predictions, dict_inputs, dict_audit_inputs, conn)
    
    
    lg.info(log_name_main,'Finalizando ejecución esquema train...')

    
main()