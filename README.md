# dvc-exp-mnist

### Get started

To get started, clone this repository and navigate to it.

The only other prerequisite is [docker](https://www.docker.com/). Once docker is installed, build a
docker image from the existing Dockerfile and run it:

```bash
docker build -t dvc-exp-mnist .
docker run -v $(pwd):/home/jovyan/dvc-exp-mnist -p 8888:8888 $(docker images -q dvc-exp-mnist)
```

To run the notebook, navigate to the link provided in the output that starts with http://127.0.0.1:8888/.
