import keras
from optkeras.optkeras import OptKeras
import tensorflow as tf
import psycopg2
from io import StringIO
import sys
import pandas as pd



class Optimize:
    """
    Clase creada para utilizar optkeras (basado en Optuna) para bucar los mejores 
    hiperparámetros de un modelo.
    """

    _conf_optuna = None
    optuna = None
    _db = None
    _user_db=None
    _pass_db=None
    _host_db=None


    def __init__(self, project="dummy", project_db="mgm_optuna", 
        user_db="postgres", pass_db="postgres", host_db="172.0.0.1", direction= "maximize"):
        """
        Constructor de la clase. Establece los parámetros para la inicialización  de OptKeras.
        
        Args:
            project (str): Estudio a realizar.
            
            project_db (str): BBDD a usar para almacenar los estudios.
            
            user_db (str): Usuario de la BBDD.
            
            pass_db (str): Contraseña de la BBDD.
            
            host_db (str): Servidor de la BBDD.
            
            direction (str): Si necesitamos maximizar (maximize) o minimizar (minimize) una métrica.

        """
        self._conf_optuna = {
        'study_name' : project,
        'storage' : 'postgresql://{}:{}@{}/{}'.format(user_db, pass_db, host_db, project_db) ,
        'load_if_exists' : True,
        'direction' : direction,
        'enable_keras_log': False,
        'enable_optuna_log': False,  
        'enable_pruning': True ,
        'verbose' :0
        }
        self._db = project_db
        self._user_db=user_db
        self._pass_db = pass_db
        self._host_db = host_db

        self.optuna = OptKeras(**self._conf_optuna)


        
        
    def run_optuna(self, function, timeout=None):
        """
        Correr el proceso de optuna. 

        Args:

            function (function): Función a maximizar o minimizar.
            
            timeout(int): Número de segundos de ejecución.  
        """
        self.optuna.optimize(function, timeout = timeout) 
        
    def get_best_params(self):
        """
        Obtener los mejores parámetros de un estudio.
        """
        return self.optuna.study.best_params
    def get_best_acc(self):
        """
        Obtener la mejor precisión de un estudio.
        """
        return self.optuna.study.trials_dataframe().user_attrs.val_acc.max()
    
    def get_best_trial(self):
        """
        Obtener el id del mejor trial de un estudio.
        """
        return self.optuna.study.best_trial[0]
    
    
    
    def _get_best_trial_acc(self, study_id):
        """
        Obtener el mejor id de trial y el mejor accuracy de un estudio.
        """
        conn = psycopg2.connect(host=self._host_db,database=self._db, user=self._user_db, password=self._pass_db)
        cur = conn.cursor()
        cur.execute("""SELECT trial_id, value
                        FROM trials 
                        WHERE study_id={} 
                        AND state  IN ('COMPLETE')
                        AND value <= 1
                        ORDER BY value desc
                        LIMIT 1""".format(study_id))
        trials= cur.fetchone()
        cur.close()
        conn.close()
        if (trials and len(trials) > 1):
            return (trials[0], trials[1])
        return (None, None)
    
    def _get_trials_time(self, study_id):
        """
        Método privado para obtener el tiempo de ejecución total de un estudio. Descarta los trials fallidos o en ejecución.

        Args:
            study_id (int): ID del estudio.

        """

        conn = psycopg2.connect(host=self._host_db,database=self._db, user=self._user_db, password=self._pass_db)
        cur = conn.cursor()
        cur.execute("""SELECT  COUNT(*) n_trials, SUM(datetime_complete-datetime_start) as time 
                    from trials 
                    WHERE state NOT IN ('FAIL', 'RUNNING') AND study_id = """ + str(study_id))
        trials= cur.fetchone()
        cur.close()
        conn.close()
        if (trials and len(trials) > 1):
            return (trials[0], trials[1])
        return (None, None)
    
    def _get_params(self, trial_id):
        """
        Obtener los parámetros de un trial concreto.
        """
        conn = psycopg2.connect(host=self._host_db,database=self._db, user=self._user_db, password=self._pass_db)
        cur = conn.cursor()
        cur.execute("""SELECT param_name, param_value 
                        FROM trial_params 
                        WHERE trial_id =""" + str(trial_id))
        
        params = {(k,v) for (k,v) in cur}
        cur.close()
        conn.close()
        return params
        
        
        
        
    def _get_study_data(self, project, study_id):
        """
        Método privado para obtener datos de un estudio. 

        Args:
            project (str): Estudio del que obtener los datos. 

        Returns
            (float, dict, int, time, int): Precición, diccionario con mejores parámetros, 
            número de intentos, tiempo total de ejecución, id del mejor trial.
        """
        conf_optuna = {
        'study_name' : project,
        'storage' : 'postgresql://{}:{}@{}/{}'.format(self._user_db, self._pass_db, self._host_db, self._db),
        'load_if_exists' : True,
        'direction' : 'maximize',
        'enable_keras_log': False,
        'enable_optuna_log': False,  
        'enable_pruning': True ,
        'verbose' :0
        }
        #opt = OptKeras(**conf_optuna)
        #try:
        import logging
        
        (btrial, acc) = self._get_best_trial_acc(study_id)
        if (not (acc and acc <= 1)):
            return (None, None, None, None, None)
            
        bp = self._get_params(btrial)
        #bp = opt.study.best_params
        #acc = opt.study.trials_dataframe().user_attrs.val_acc.max()
        #btrial =  opt.study.best_trial[0]
        (ntrials,time) = self._get_trials_time(study_id)
        return (acc, bp,ntrials, time, btrial)
        
        #except:
        #    return (None, None, None, None, None)
        
        
    def get_studies(self): 
        """
        Obtener los estudios realizados.

        Returns:
            dataframe: Dataframe con los datos de todos los estudios.
        """
        aux_stdout = sys.stdout
        sys.stdout = StringIO()
        conn = psycopg2.connect(host=self._host_db,database=self._db, user=self._user_db, password=self._pass_db)
        cur = conn.cursor()
        cur.execute('SELECT study_id, study_name FROM studies')
        studies =[]
        #study= cur.fetchone()
        for study in cur:
            (acc, bp, ntrials, time, btrial ) = self._get_study_data(study[1], study[0])
            if(acc and bp):
                studies.append((study[1], acc, bp, ntrials, time, btrial))
            #study=cur.fetchone()
        cur.close()
        conn.close()
        sys.stdout = aux_stdout
        
        df = pd.DataFrame(studies, columns=['study', 'accuracy', 'params', "ntrials", "total_time", "best_trial"])

        return df
    
    def delete_study(self, study):  
        """
        Eliminar un estudio y sus trials de la base de datos.

        Args:
            study (str): Nombre del estudio a eliminar.
        """
        conn = psycopg2.connect(host=self._host_db,database=self._db, user=self._user_db, password=self._pass_db)
        cur = conn.cursor()
        cur.execute("SELECT * FROM studies WHERE study_name = '" + study +"'")
        studies =[]
        study= cur.fetchone()
        if (study):
            id_study = study[0]
            tables_study = ["studies", "study_system_attributes", "study_user_attributes"]
            tables_trial = ["trial_system_attributes", "trial_user_attributes", "trial_values", "trial_params", "trials"]
            
    
            #get trials ids
            cur.execute("SELECT DISTINCT(trial_id) FROM trials WHERE study_id = " + str(id_study))
            trials = [t[0] for t in cur]
            for trial_id in trials: 
                for t in tables_trial: 
                    cur.execute("DELETE FROM " + t +  " WHERE trial_id = " + str(trial_id))
            for t in tables_study: 
                cur.execute("DELETE FROM " + t +  " WHERE study_id = " + str(id_study))
            print (id_study)
        cur.close()
        conn.commit()
        conn.close()
        
    def rename_study(self, study, new_name):
        """
        Renombrar estudio.

        Args:
            study (str): Nombre original del estudio.

            new_name (str): Nuevo nombre del estudio.

        """
        conn = psycopg2.connect(host=self._host_db,database=self._db, user=self._user_db, password=self._pass_db)
        cur = conn.cursor()
        cur.execute("UPDATE studies SET study_name = '" + new_name + "' WHERE study_name = '" + study +"'")        
        cur.close()
        conn.commit()
        conn.close()
