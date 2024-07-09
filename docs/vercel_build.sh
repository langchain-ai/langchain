#!/bin/bash

set -e

make install-vercel-deps

make build

rm -rf docs
mv build/output-new/docs ./
