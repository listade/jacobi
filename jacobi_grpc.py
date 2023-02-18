# Generated by the Protocol Buffers compiler. DO NOT EDIT!
# source: jacobi.proto
# plugin: grpclib.plugin.main
import abc
import typing

import grpclib.const
import grpclib.client
if typing.TYPE_CHECKING:
    import grpclib.server

import jacobi_pb2


class JacobiCalcBase(abc.ABC):

    @abc.abstractmethod
    async def Calc(self, stream: 'grpclib.server.Stream[jacobi_pb2.Request, jacobi_pb2.Result]') -> None:
        pass

    def __mapping__(self) -> typing.Dict[str, grpclib.const.Handler]:
        return {
            '/JacobiCalc/Calc': grpclib.const.Handler(
                self.Calc,
                grpclib.const.Cardinality.UNARY_UNARY,
                jacobi_pb2.Request,
                jacobi_pb2.Result,
            ),
        }


class JacobiCalcStub:

    def __init__(self, channel: grpclib.client.Channel) -> None:
        self.Calc = grpclib.client.UnaryUnaryMethod(
            channel,
            '/JacobiCalc/Calc',
            jacobi_pb2.Request,
            jacobi_pb2.Result,
        )
