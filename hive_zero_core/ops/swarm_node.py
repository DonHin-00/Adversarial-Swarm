import grpc
from concurrent import futures
import logging
from hive_zero_core.utils import swarm_pb2, swarm_pb2_grpc

class SwarmNode(swarm_pb2_grpc.SwarmServiceServicer):
    """
    gRPC Server for Distributed HiveMind Synchronization.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.knowledge_store = []
        self.logger = logging.getLogger(f"SwarmNode-{node_id}")

    def ShareKnowledge(self, request, context):
        self.logger.info(f"Received update from {request.source_id}")
        self.knowledge_store.append(request.embedding)
        return swarm_pb2.Ack(success=True)

    def SyncState(self, request, context):
        # Aggregate stored knowledge (Mock mean)
        if self.knowledge_store:
            # Assume embeddings are serialized tensors?
            # Simplified: just return dummy bytes
            agg = b"aggregated_data"
        else:
            agg = b""
        return swarm_pb2.GlobalState(aggregated_embedding=agg, active_experts=10)

    def start_server(self, port: int = 50051):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        swarm_pb2_grpc.add_SwarmServiceServicer_to_server(self, server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        self.logger.info(f"Swarm Node started on port {port}")
        return server
