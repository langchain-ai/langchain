#!/bin/bash


yum -y update
yum install gcc bzip2-devel libffi-devel zlib-devel wget tar gzip -y

wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.3.450/quarto-1.3.450-linux-amd64.tar.gz
tar -xzf quarto-1.3.450-linux-amd64.tar.gz

set -e

QUARTO_CMD="./quarto-1.3.450/bin/quarto" make build
