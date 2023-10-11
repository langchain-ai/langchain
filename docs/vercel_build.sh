#!/bin/bash

version_compare() {
    local v1=(${1//./ })
    local v2=(${2//./ })
    for i in {0..2}; do
        if (( ${v1[i]} < ${v2[i]} )); then
            return 1
        fi
    done
    return 0
}

openssl_version=$(openssl version | awk '{print $2}')
required_openssl_version="1.1.1"

python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_python_version="3.10"

echo "OpenSSL Version"
echo $openssl_version
echo "Python Version"
echo $python_version
# If openssl version is less than 1.1.1 AND python version is less than 3.10
if ! version_compare $openssl_version $required_openssl_version && ! version_compare $python_version $required_python_version; then
### See: https://github.com/urllib3/urllib3/issues/2168
# Requests lib breaks for old SSL versions,
# which are defaults on Amazon Linux 2 (which Vercel uses for builds)
    yum -y update
    yum remove openssl-devel -y
    yum install gcc bzip2-devel libffi-devel zlib-devel wget tar -y
    yum install openssl11 -y
    yum install openssl11-devel -y

    wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz
    tar xzf Python-3.11.4.tgz
    cd Python-3.11.4
    ./configure
    make altinstall
    echo "Python Version"
    python3.11 --version
    cd ..
fi

python3.11 -m venv .venv
source .venv/bin/activate
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r vercel_requirements.txt
python3.11 scripts/model_feat_table.py
nbdoc_build --srcdir docs
python3.11 scripts/generate_api_reference_links.py
