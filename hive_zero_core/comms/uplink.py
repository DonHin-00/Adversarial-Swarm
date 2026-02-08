from fastapi import APIRouter, Request
from pydantic import BaseModel
import logging

uplink_router = APIRouter()
logger = logging.getLogger("LarvaUplink")

class ExfilData(BaseModel):
    data: str

@uplink_router.post("/uplink")
async def receive_uplink(data: ExfilData):
    """
    Dead-drop receiver for Larva variants.
    Unidirectional: Accepts data, returns generic 200 OK.
    """
    logger.info(f"Received Larva Exfil: {len(data.data)} bytes")
    # Store in Evidence DB or Graph
    # For now, just log
    return {"status": "ack"}
