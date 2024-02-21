# Using Text Generation Inference as Serving framework on Habana Gaudi

To use [🤗 text-generation-inference](https://github.com/huggingface/text-generation-inference) on Habana Gaudi/Gaudi2, please follow these steps:

## Build the Docker image located in the tgi-gaudi repo:
```bash
git clone https://github.com/huggingface/tgi-gaudi.git
cd tgi-gaudi/
docker build -t tgi_gaudi .
```

If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` like below.
```bash
docker build -t tgi_gaudi . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy

## Launch a local server instance on 1 Gaudi card:
```bash
model=Intel/neural-chat-7b-v3-3
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tgi_gaudi --model-id $model
```

If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` like below.
```bash
docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -e HTTPS_PROXY=$https_proxy -e HTTP_PROXY=$https_proxy tgi_gaudi --model-id $model
```

For gated models such as LLAMA-2, you will have to pass -e HUGGING_FACE_HUB_TOKEN=<token> to the docker run command above with a valid Hugging Face Hub read token.

Send a request to check if the endpoint is wokring:

```bash
curl localhost:8080/generate   -X POST   -d '{"inputs":"Which NFL team won the Super Bowl in the 2010 season?","parameters":{"max_new_tokens":128, "do_sample": true}}'   -H 'Content-Type: application/json'
```
The first call will be slower as the model is compiled.

More details please refer to [tgi-gaudi](https://github.com/huggingface/tgi-gaudi/blob/v1.2-release/README.md).

For more information and documentation about Text Generation Inference, checkout the [README](https://github.com/huggingface/text-generation-inference#text-generation-inference) of the original repo.


## Install intel optimized langchain
```bash
git clone https://github.com/lvliang-intel/intel_genai_kit_langchain.git
cd intel_genai_kit_langchain/libs/langchain/
pip install -e .
cd ../community/
pip install -e .
```

## Install huggingface_hub
```bash
pip install huggingface_hub
```

## Access Service
```bash
cd ../examples/serving/tgi/
export HUGGINGFACEHUB_API_TOKEN=<token>
python client.py
```
