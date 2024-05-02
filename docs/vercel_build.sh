#!/bin/bash

set -e

make install-vercel-deps

QUARTO_CMD="./quarto-1.3.450/bin/quarto" make build-new

rm -rf docs
mv build/output-new/docs ./
