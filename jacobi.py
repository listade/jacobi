import argparse
import asyncio
from math import ceil

import numpy as np

from grpclib.client import Channel
from grpclib.server import Server

from jacobi_pb2 import Request, Result
from jacobi_grpc import JacobiCalcStub, JacobiCalcBase


class JacobiCalc(JacobiCalcBase):

    async def Calc(self, stream):
        msg = await stream.recv_message()

        A = np.frombuffer(msg.A)
        B = np.frombuffer(msg.B)
        x = np.frombuffer(msg.x)
        
        off = int(msg.offset)

        n = len(B)
        m = len(A) // n 

        A = A.reshape((n, m))

        x = [1 / A[i, i + off] * (B[i] - np.sum([A[i, j] * x[j] for j in range(m) if (i + off) != j])) for i in range(n)]
        x = np.array(x)

        await stream.send_message(Result(x=x.tobytes()))



async def run_client():
    data = np.loadtxt(opt.data, ndmin=2, dtype=np.float64, delimiter=',')

    A = data[:,:-1]
    B = data[:,-1:].flatten()

    N, _ = A.shape
    x = np.zeros(N)

    eps = 1e-3

    nodes = list(map(lambda v: v.split(':'), opt.nodes))
    channels = [Channel(*v) for v in nodes]
    nodes_count = len(nodes)
    block_len = ceil(N / nodes_count)
    blocks = ceil(N / block_len)

    K = 1000

    for i in range(K):
        tasks = list()
        
        for j in range(blocks):
            a = j * block_len
            b = a + block_len

            block_A = A[a:b,:]
            block_B = B[a:b]

            msg = Request(A=block_A.tobytes(), 
                          B=block_B.tobytes(),
                          offset=a, 
                          x=x.tobytes())

            stub = JacobiCalcStub(channels[j % len(channels)])
            tasks.append(stub.Calc(msg))
        
        x = np.concatenate([np.frombuffer((await v).x) for v in tasks], axis=0)
        K = np.abs(A.dot(x) - B)
        
        if np.max(K) <= eps:
            print(i, x)
            break

    for v in channels:
        v.close()


async def run_server():
    server = Server([JacobiCalc()])

    await server.start('0.0.0.0', int(opt.port))
    await server.wait_closed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    role_group = parser.add_mutually_exclusive_group()

    role_group.add_argument('--client', action='store_true')
    role_group.add_argument('--server', action='store_true')

    parser.add_argument('--data', type=str, default='data.csv')
    parser.add_argument('--nodes', type=str, default=['127.0.0.1:5000'], nargs='+')
    parser.add_argument('--port', type=int, default=5000)

    opt = parser.parse_args()

    if opt.client:
        asyncio.run(run_client())
    elif opt.server:
        asyncio.run(run_server())
