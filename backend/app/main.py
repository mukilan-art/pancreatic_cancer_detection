from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import time
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.services.analysis_service import AnalysisService
from app.api.routes import health, analyze, models_status, segment, classify, gradcam, report, preprocess, diagnostics

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    service = AnalysisService(settings)
    service.initialize()
    yield
    # Cleanup here if needed

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan
)

origins = settings.CORSORIGINS.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_code": "INTERNAL_ERROR"}
    )

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(analyze.router, prefix="/api", tags=["analyze"])
app.include_router(models_status.router, prefix="/api", tags=["models"])
app.include_router(segment.router, prefix="/api", tags=["segment"])
app.include_router(classify.router, prefix="/api", tags=["classify"])
app.include_router(gradcam.router, prefix="/api", tags=["gradcam"])
app.include_router(report.router, prefix="/api", tags=["report"])
app.include_router(preprocess.router, prefix="/api", tags=["preprocess"])
app.include_router(diagnostics.router, prefix="/api", tags=["diagnostics"])

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to Pancreatic Cancer XAI API"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG,
    )
