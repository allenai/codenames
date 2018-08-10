#!/usr/bin/env bash

set -x
set -e

wget -nc https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.06.txt.gz -P data/
tar -xvzf data/numberbatch-en-17.06.txt.gz