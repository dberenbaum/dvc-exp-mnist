FROM jupyter/base-notebook

# Enable Jupyter Lab
ENV JUPYTER_ENABLE_LAB=yes

# Install dependencies.
USER root
RUN apt-get update && apt-get install -yq git
COPY --chown=${NB_UID}:${NB_GID} environment.yaml /tmp/
RUN conda env update -n base -f /tmp/environment.yaml && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
    FROM jupyter/base-notebook
