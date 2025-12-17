from fastapi import APIRouter

from app.api import pipeline, data, chart, edge_finder

api_router = APIRouter()

api_router.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])
api_router.include_router(data.router, prefix="/instruments", tags=["instruments"])
api_router.include_router(chart.router, prefix="/chart", tags=["chart"])
api_router.include_router(edge_finder.router, prefix="/edge-finder", tags=["edge-finder"])
