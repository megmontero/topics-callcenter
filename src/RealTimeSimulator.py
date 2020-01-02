#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from time import sleep


# In[1]:


from confluent_kafka import Producer

def delivery_report(err, msg):
    if err is not None:
        print("Failed: {}".format(err))
   # else:
   #     print("Delivered to {} [{}]".format(msg.topic(), msg.partition()))



producer_conf = {"bootstrap.servers": "kfk1:9093",
                "security.protocol": "SASL_PLAINTEXT",
                "sasl.mechanism": "SCRAM-SHA-256",
                 "sasl.username": "topic_model",
                 "sasl.password": "password"
                }
producer = Producer(producer_conf)


# In[3]:


verint_raw = pd.read_parquet('/data/datasets/input_data/verint_dataset.parquet')


# In[132]:


ivr_hierarchy = pd.read_excel('/data/mgm/data/dwproyp0.dwen_1004_14_etiquetas_23102019.xlsx', index_col=0, header=0)
last_date = ivr_hierarchy["fx_carga"].max()
print("Nos quedamos con la fecha: {}".format(last_date))
ivr_hierarchy = ivr_hierarchy[ivr_hierarchy["fx_carga"] == last_date].drop(columns="fx_carga")
ivr_hierarchy.columns = ["tipo", "subtipo", "no_destino_pa"]


# Vamos a sacar como se distribuyen las llamadas a lo largo del día

# In[69]:


pd.set_option('max_rows',9999)
pd.set_option('max_columns', 9999)
pd.set_option('display.max_colwidth', 500) #-1 unlimited
verint = verint_raw[["co_llamada_verint","fx_evento", "audio_start_time","cd22", "duration_seconds", "no_destino_pa", "plaintext" ]]


prov_spain={ 'La Coruna': 'ES-C',
 'Alava': 'ES-VI',
 'Albacete': 'ES-AB',
 'Alicante': 'ES-A',
 'Almeria': 'ES-AL',
 'Asturias': 'ES-O',
 'Avila': 'ES-AV',
 'Badajoz': 'ES-BA',
 'Baleares': 'ES-PM',
 'Barcelona': 'ES-B',
 'Burgos': 'ES-BU',
 'Caceres': 'ES-CC',
 'Cadiz': 'ES-CA',
 'Cantabria': 'ES-S',
 'Castellon': 'ES-CS',
 'Ciudad Real': 'ES-CR',
 'Cordoba': 'ES-CO',
 'Cuenca': 'ES-CU',
 'Gerona': 'ES-GI',
 'Granada': 'ES-GR',
 'Guadalajara': 'ES-GU',
 'Guipuzcoa': 'ES-SS',
 'Huelva': 'ES-H',
 'Huesca': 'ES-HU',
 'Jaen': 'ES-J',
 'La Rioja': 'ES-LO',
 'Las Palmas': 'ES-GC',
 'Leon': 'ES-LE',
 'Lerida': 'ES-L',
 'Lugo': 'ES-LU',
 'Madrid': 'ES-M',
 'Malaga': 'ES-MA',
 'Murcia': 'ES-MU',
 'Navarra': 'ES-NA',
 'Orense': 'ES-OR',
 'Palencia': 'ES-P',
 'Pontevedra': 'ES-PO',
 'Salamanca': 'ES-SA',
 'Santa Cruz de Tenerife': 'ES-TF',
 'Segovia': 'ES-SG',
 'Sevilla': 'ES-SE',
 'Soria': 'ES-SO',
 'Tarragona': 'ES-T',
 'Teruel': 'ES-TE',
 'Toledo': 'ES-TO',
 'Valencia': 'ES-V',
 'Valladolid': 'ES-VA',
 'Vizcaya': 'ES-BI',
 'Zamora': 'ES-ZA',
 'Zaragoza': 'ES-Z',
  'Ceuta': "ES-CE",
  'Melilla': "ES-ML"}
    
def get_province_code(prov):
    if len(prov) == 0 or "Desconocido" in prov:
        return "ES-UNK"
    prov= prov.split("-")[1]
    
    
    return prov if prov not in prov_spain else prov_spain[prov]
    
    
    
    
    
    return prov

verint["cod_prov"] = verint["cd22"].apply(get_province_code)


# In[135]:


verint =  pd.merge(verint, ivr_hierarchy, on=['no_destino_pa'], how='inner')


# In[104]:


verint["day"] =  verint["audio_start_time"].apply(lambda x: str(x.year * 10000 + x.month * 100 + x.day ))

call_per_day = (verint.groupby(['day']).count().reset_index()[['day', 'co_llamada_verint']]).set_index(["day"])
total_calls = call_per_day.sum()["co_llamada_verint"]
call_per_day.describe()


# In[108]:


verint["hour"] =  verint["audio_start_time"].apply(lambda x: str(x.hour).zfill(2))
call_per_hour = (verint.groupby(['hour']).count().reset_index()[['hour', 'co_llamada_verint']]).set_index(["hour"])
percents = list(call_per_hour["co_llamada_verint"].apply(lambda x: x/total_calls))


# In[119]:


# Como transcribimos solo el 4% multiplicamos por 25
from math import ceil, floor
total_calls_day = int(call_per_day.quantile(0.9)* 25)
calls_per_min = [ceil(x*total_calls_day/60) for x in percents]


# In[131]:


from datetime import datetime

total_calls_day


# In[129]:


#Flujo para transmitir 
import json

len_verint = len(verint)
idx = 0
import random
def publish_call(ncalls):
    global idx, len_verint, verint
    producer.poll(0)
    for i in range(ncalls):
        idx = (idx + 1)%len_verint

        co_llamada = verint.iloc[idx]["co_llamada_verint"]
        msg = {
            "call_text": str(verint.iloc[idx]["plaintext"]),
            "co_verint": str(co_llamada),
            "call_timestamp": int(datetime.now().timestamp()),#verint.iloc[idx]["audio_start_time"],
            "duration": str(verint.iloc[idx]["duration_seconds"]),
            "province": str(verint.iloc[idx]["cd22"]),
            "co_province": str(verint.iloc[idx]["cod_prov"])
        }
        
        if ((idx%3)==0):#metemos llamadas de control
            msg["control_type"] = verint.iloc[idx]["tipo"];
        
       # msg = '{{"call_text":"{}", "co_verint": "{}", "call_timestamp": "{}", "province": "{}", "co_province": "{}", "duration": {} }}'\
       # .format(text,co_llamada,start_time, province, co_province, duration )\
       # .encode('utf-8')
        producer.produce("CALLS",json.dumps(msg) ,co_llamada, callback=delivery_report)
    producer.flush()
    


    
def start_rt_simulator():
    while(True):
        llamadas_min = calls_per_min[datetime.now().hour]
        if (llamadas_min < 60):
            publish_call(llamadas_min)
            sleep(60)
        elif (llamadas_min > 60):
            llamadas_sec = random.randint(floor(llamadas_min/60.0),ceil(llamadas_min/60.0))
            publish_call(llamadas_sec)
            sleep(1)
        



# In[130]:


start_rt_simulator()

