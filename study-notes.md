### 需要安装pip
sudo apk add py3-pip

### 需要建立虚拟环境
```
python -m venv .venv
# 启动虚拟环境
. .venv/bin/activate

pip install poetry
poetry shell

# 安装ipykernel
sudo apk add gcc python3-dev musl-dev linux-headers
python -m pip install ipykernel -U --force-reinstall
```