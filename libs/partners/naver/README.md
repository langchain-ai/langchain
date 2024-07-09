# langchain-naver

All functionality related to CLOVA X, the AI technology ecosystem of Naver and Naver Cloud, especially via [CLOVA Studio](https://clovastudio.ncloud.com/).

> [Naver](https://navercorp.com/) is a global technology company based in South Korea with cutting-edge technologies and a diverse business portfolio including search, commerce, fintech, content, cloud, and AI.

> [Naver Cloud](https://www.navercloudcorp.com/lang/en/) is the cloud computing arm of Naver, a leading cloud service provider offering a comprehensive suite of cloud services to businesses through its Naver Cloud Platform (NCP).

Please refer to [NCP User Guide](https://guide.ncloud-docs.com/docs/clovastudio-overview) for more detailed instructions (also in Korean).

## Installation and Setup

- Get both CLOVA Studio API Key and API Gateway Key by [creating your app](https://guide.ncloud-docs.com/docs/en/clovastudio-playground01#create-test-app) and set them as environment variables respectively (`NCP_CLOVASTUDIO_API_KEY`, `NCP_APIGW_API_KEY`).
- Install the integration Python package with:

```bash
pip install -U langchain-naver
```

Get both CLOVA Studio API Key and API Gateway Key by [creating your app](https://guide.ncloud-docs.com/docs/en/clovastudio-playground01#create-test-app) and set them as environment variables respectively (`NCP_CLOVASTUDIO_API_KEY`, `NCP_APIGW_API_KEY`).

## Chat Models

### ChatClovaX

See a [usage example](/docs/integrations/chat/naver).

```python
from langchain_naver import ChatClovaX
```

## Embedding Models

### ClovaXEmbeddings

See a [usage example](/docs/integrations/text_embedding/naver).

```python
from langchain_naver import ClovaXEmbeddings
```
