FROM jupyter/base-notebook

# Install git
USER root
RUN apt-get update && apt-get install -yq git
RUN git clone https://github.com/iterative/dvc.git
RUN cd dvc && pip install -e .

# Clone repo and update dependencies
USER jovyan
RUN git clone https://github.com/dberenbaum/dvc-exp-mnist.git
WORKDIR dvc-exp-mnist
RUN git checkout 2.0
RUN conda env update -n base -f environment.yaml
