#!/bin/bash

#List pycache dirs inside megatron and subdirs therof and remove them
find megatron -name "__pycache__" -ok rm -rf {} \;