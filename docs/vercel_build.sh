#!/bin/bash

set -e

make install-vercel-deps

make build-new

rm -rf docs
mv build/output-new/docs ./
