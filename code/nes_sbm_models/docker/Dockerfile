FROM pytorch/pytorch:latest
RUN apt-get update && \
    apt-get install -qqy graphviz libgraphviz-dev graphviz-dev pkg-config
RUN pip install --upgrade pip
RUN pip install networkx tables pandas matplotlib jupyter pygraphviz
RUN pip install notebook
RUN apt-get install -y rsync

