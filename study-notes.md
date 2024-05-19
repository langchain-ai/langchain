### 需要安装pip
sudo apk add py3-pip

### 需要建立虚拟环境
python -m venv .venv
. .venv/bin/activate
pip install poetry
poetry shell
sudo apk add gcc python3-dev musl-dev linux-headers
python -m pip install ipykernel -U --force-reinstall