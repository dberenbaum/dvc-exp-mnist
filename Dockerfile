FROM jupyter/base-notebook

# Install git
USER root
RUN apt-get update && apt-get install -yq git

# Clone repo and update dependencies
USER jovyan
RUN git clone https://github.com/dberenbaum/dvc-exp-mnist.git
WORKDIR dvc-exp-mnist
RUN git checkout 2.0
RUN conda env update -f environment.yaml
