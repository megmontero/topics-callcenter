# Capa de servicio - ELK

En este apartado se presentan los ficheros necesarios  para la configuración de la ingesta, visualizaciones y alarmado en el stack de Elastic.

En este apartado nos encontramos:  


- **kibana_objects.ndjson**: Objetos de Kibana: Visualizaciones, Dashboards e index patterns.
- **templates.yml**: Plantillas de los índices de Elasticsearch. 
- **pipeline.yml**: Pipeline de ingesta en Logstash.
- **watcher.yml**: Alarma estática de recursos para Elasticsearch.
- **anomaly_detector.yml**: Job de machine learning para detectar anomalias.


