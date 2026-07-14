from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import router as api_router

app = FastAPI(
    title="Product Recommendation API",
    description="Backend API for the MLOps End-to-End Product Recommendation System.",
    version="1.0.0",
)

# Configure CORS for the frontend (Phase 5)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {
        "message": "Welcome to the Product Recommendation API",
        "docs": "/docs"
    }
