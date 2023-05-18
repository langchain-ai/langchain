#!/bin/bash
set -ex

url="https://karpathy.ai/lexicap/data.zip"
filename="data.zip"
directory="transcripts"

mkdir -p $directory
curl -o $filename $url
tar -xf $filename -C $directory
rm $filename
