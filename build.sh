#!/bin/sh

python -m grpc_tools.protoc -I. --python_out=. --grpclib_python_out=. jacobi.proto
