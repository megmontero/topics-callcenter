FROM tensorflow/serving


MAINTAINER Manuel Gomez mnlgmontero@gmail.com

LABEL io.k8s.description="Tensorflow serving builder" \
      io.k8s.display-name="tensorflow serving builder" \
      io.openshift.expose-services="8501:http" \
      io.openshift.tags="tf,serving,model" \
      io.openshift.s2i.scripts-url="image:///usr/libexec/s2i"


COPY ./s2i/bin/ /usr/libexec/s2i
COPY ./container-entrypoint /usr/bin/


RUN chown -R 1001:1001 /models && mkdir -p /tmp/src/ && mkdir -p /tmp/s2i/src/ && chown -R 1001:1001 /tmp/  &&  chmod +x /usr/bin/container-entrypoint && mkdir /conf && chmod 777 /conf

COPY monitoring_config.txt /conf/

USER 1001

EXPOSE 8501


ENTRYPOINT ["container-entrypoint"]

CMD ["usage"]



