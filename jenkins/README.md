# Modelización de topics en llamadas al Call Center
En este repositorio se encuentra el código del TFM "Modelización de topics en llamadas al Call Center". Este TFM consta de dos partes. Por un lado, el entrenamiento de modelos supervisados (mediante técnicas de *deep learning*) y modelos no supervisados. Posteriormente el paso de uno de estos modelos a producción y su utilización en Real Time con la arquitectura de la siguiente imagen.

![Arquitectura kappa](images/kappa_v1.png "Arquitectura Kappa")



Además el despliegue de todo el proyecto se ha realizado en contenedores sobre Openshift, aplicando integración y despliegue continuo con Jenkins como podemos observar en la siguiente imagen. 


![CICD](images/cicd_v1.png "CICD")


En el archivo **TFM.pdf** podemos encontrar la documentación completa del proyecto, a excepción de la documentación del código fuente Python y Java que se encuentra en el directorio src/.



## Estructura Repositorio

El repositorio se encuentra estructurado del siguiente modo: 

- **latex/**:  Directorio con todo el código latex de la memoria. 
- **notebooks/**: Directorio con notebooks de Jupyter.
- **notebooks/html/**: Directorio con el export de algunos notebooks en HTML. 
- **TFM borrador.pdf**: Última versión del borrador de la memoria.
- **pecs/**: Entrega de las diferentes PECs.
- **src/**: Distintos códigos fuente
