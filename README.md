If no cuda development toolkits installed, do development inside nvidia's cuda-devel docker image:

```docker run --rm --ipc=host --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.4.1-devel-ubuntu22.04```

then install system packages:

```
sudo apt-get update && sudo apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs
```

then `pip install -r requirements.txt` and `bentoml serve .`
