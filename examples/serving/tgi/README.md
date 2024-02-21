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
model=Intel/neural-chat-7b-v3-1
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tgi_gaudi --model-id $model
```

## Launch a local server instance on 8 Gaudi cards:
```bash
model=Intel/neural-chat-7b-v3-1
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 -v $volume:/data --runtime=habana -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tgi_gaudi --model-id $model --sharded true --num-shard 8
```


If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` like below.
```bash
docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -e HTTPS_PROXY=$https_proxy -e HTTP_PROXY=$https_proxy tgi_gaudi --model-id $model
```


> Set `LIMIT_HPU_GRAPH=True` for larger sequence/decoding lengths(e.g. 300/212).

## Install intel optimized langchain
```bash
git clone https://github.com/lvliang-intel/intel_genai_kit_langchain.git
cd intel_genai_kit_langchain/libs/langchain/
pip install -e .
```

## Access Service
```bash
python client.py
```
