# Capa de servicio - ELK

En este apartado se presentan los ficheros necesarios  para la configuración de la ingesta, visualizaciones y alarmado en el stack de Elastic.

![servicio](../images/servicelayer_v1.png "Capa de servicio")


En este apartado nos encontramos:  


- **kibana_objects.ndjson**: Objetos de Kibana: Visualizaciones, Dashboards e index patterns.
- **templates.yml**: Plantillas de los índices de Elasticsearch. 
- **pipeline.yml**: Pipeline de ingesta en Logstash.
- **watcher.yml**: Alarma estática de recursos para Elasticsearch.
- **anomaly_detector.yml**: Job de machine learning para detectar anomalias.
- **config**: Directorio con ficheros de configuración de Jolokia, Metricbeat y Logstash.

La documentación completa de la capa de servicio puede encontrarse en el fichero **TFM.pdf** situado en la raíz del directorio.

# Capturas

Mostramos algunas capturas de la capa de servicio.

![CM-calls](../images/CM-calls.png "CM-calls")

![ml-factura](../images/ml-factura.png "Detección de anomalias")

![wordcloud](../images/wordcloud.png "Nube de palabras")



