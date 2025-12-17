from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router

app = FastAPI(
    title="Recursive Session Waveform Analyzer",
    description="API for forex data analysis with recursive waveform visualization",
    version="1.0.0",
)

# Configure CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Recursive Session Waveform Analyzer API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
