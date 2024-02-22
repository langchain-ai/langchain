# Using Triton Inference Server as Serving framework on Habana Gaudi

To use [Triton Inference Server](https://github.com/triton-inference-server/server) on Habana Gaudi/Gaudi2, please follow these steps:

## Create a Model Repository and Docker Image

The Triton Inference Server is launched inside a docker container. The first step is to create a model repository which will be used by Triton to load your models. You can find a comprehensive guide in the following [GitHub Repository](https://github.com/triton-inference-server/server).

### Model Repository
We have created a model repository in this directory according to the structure detailed in Setting up the [model repository](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment#setting-up-the-model-repository) and [Model repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md).

You can modify the source code according to your requirement.
And You can change the num of `count` here to configure the num of model instances on your HPU, 8 is set in the example config file like below.
```
instance_group [{ 
    count: 8
    kind: KIND_CPU 
}]
```

Then the folder structure under the current `triton` folder like this:

```
triton/
├── models
│   └── text_generation
│       ├── 1
│       │   ├── model.py
│       │   ├── utils.py
│       └── config.pbtxt
├── README.md
```

## Create Docker Image for HPU
Followting the commands below, you will create a Docker image for Habana Gaudi on your local machine.

```bash
git clone https://github.com/HabanaAI/Setup_and_Install.git
cd Setup_and_Install/dockerfiles/triton
make build DOCKER_CACHE=true
```

## Run the Backend Container
After the Docker Image is created, you need to run a backend container to run tritonserver. The serving scripts will be mounted into Docker container using `-v ./models:/models`.

Remember to replace the `${image_name}` into the docker image name you just created. You can check the image name with the command `docker images`.
```bash
docker run -it --runtime=habana --name triton_backend --shm-size "4g" -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v ./models:/models ${image_name}
```

If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` like below.
```bash
docker run -it --runtime=habana --name triton_backend --shm-size "4g" -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v ./models:/models --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy ${image_name}
```


## Launch the Triton Server
Now you should be inside the Docker container.

In order to launch your customized triton server, you need to install the necessary prerequisites for itrex. By default, NeuralChat uses `Intel/neural-chat-7b-v3-3` as the LLM. Then you can launch the triton server to start the service.

You can specify an available http port to replace the `${your_port}` in tritonserver command.
```bash
# launch triton server
tritonserver --model-repository=/models --http-port ${your_port}
```

When the triton server is successfully launched, you will see the table below:
```bash
I0103 00:04:58.435488 237 server.cc:626]
+--------+---------+--------+
| Model  | Version | Status |
+--------+---------+--------+
| llama2 | 1       | READY  |
+--------+---------+--------+
```

Check the service status and port by running the following command:
```bash
curl -v localhost:8021/v2/health/ready
```

You will find a `HTTP/1.1 200 OK` if your server is up and ready for receiving requests.


## Launch and Run the Client

Start the Triton client and enter into the container. Remember to replace the `${image_name}` into the docker image name you just created.

```bash
docker run -it --runtime=habana --name triton_client --shm-size "4g" -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v ./models:/models ${image_name}
```

Inside the client docker container, you need to install tritonclient first.
```bash
pip install tritonclient[all]
```

Send a request using `client.py`. The `${your_port}` is the triton server port.
```bash
python /models/text_generation/1/client.py --prompt="Tell me about Intel Xeon Scalable Processors." --url=localhost:${your_port}
```
