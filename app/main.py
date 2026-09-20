import asyncio
import logging
import os
from contextlib import asynccontextmanager

# Auto-detect hardware and configure before any model loading.
# Explicit env vars (e.g. for production) always override autoconfig.
from app.autoconfig import run_autoconfig, format_banner, get_device

_hw, _profile, _config = run_autoconfig()

import torch

torch.set_num_threads(int(os.getenv("SLOPTOTAL_TORCH_THREADS", "1")))
torch.set_num_interop_threads(1)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import (
    init_database,
    check_database_health,
    purge_invalid_cached_reports,
    purge_expired_reports,
    REPORT_RETENTION_DAYS,
)
from app.analyzer import get_engine_list, shutdown_analyzer, _max_full, _max_snippet
from app.queue_manager import QueueManager

from app.routes.api import router as api_router
from app.routes.queue import router as queue_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sloptotal")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    print(format_banner(_hw, _profile, _config))
    log.info(f"Starting ORIGYN API (profile={_profile}, device={get_device()})...")
    await init_database()
    purged = await purge_invalid_cached_reports()
    if purged:
        log.info(f"Purged {purged} stale cached report(s) with engine load failures")

    # Enforce the retention window promised on /privacy, at boot and then daily.
    expired = await purge_expired_reports()
    if expired:
        log.info(f"Purged {expired} report(s) older than {REPORT_RETENTION_DAYS} days")
    retention_task = asyncio.create_task(_retention_loop())

    _preload_models()

    queue_manager = QueueManager(
        max_snippet=_max_snippet,
        max_quick=_max_snippet,
        max_full=_max_full,
    )
    await queue_manager.start()
    app.state.queue_manager = queue_manager

    log.info("ORIGYN API started")
    yield
    log.info("Shutting down ORIGYN API...")
    retention_task.cancel()
    await queue_manager.stop()
    shutdown_analyzer()
    log.info("ORIGYN API shutdown complete")


app = FastAPI(
    title="ORIGYN API",
    description="ORIGYN - AI Content Detection API running 23 independent detection engines",
    lifespan=lifespan,
)

cors_origins_env = os.getenv("ORIGYN_CORS_ORIGINS", os.getenv("SLOPTOTAL_CORS_ORIGINS", "*"))
cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if "*" not in cors_origins else ["*"],
    allow_credentials=True if "*" not in cors_origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.queue_manager = None  # set in lifespan

# API Routers
app.include_router(api_router)
app.include_router(queue_router)


@app.get("/")
async def root():
    """API entry point returning service status and available endpoints."""
    return {
        "service": "ORIGYN API",
        "description": "ORIGYN - High-accuracy AI detection API running 23 independent detection engines",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "quick_score": "POST /api/quick-score",
            "analyze": "POST /api/analyze",
            "start_analysis": "POST /api/analyze/start",
            "stream_results": "GET /api/stream/{report_id}",
            "report": "GET /api/report/{report_id}",
            "paragraph_score": "POST /api/paragraph-score",
            "scan_snippets": "POST /api/scan/snippets",
            "scan_urls": "POST /api/scan/urls",
            "engines": "GET /api/engines",
            "queue_status": "GET /api/queue/status",
        },
    }


async def _retention_loop():
    """Re-run the retention purge once a day for as long as the app is up.

    A cron entry on the host would not survive a container rebuild, and this
    process is already long-lived, so the schedule lives here. Errors are logged
    and swallowed: a failed purge must never take the API down.
    """
    while True:
        try:
            await asyncio.sleep(24 * 60 * 60)
            expired = await purge_expired_reports()
            if expired:
                log.info(
                    f"Purged {expired} report(s) older than {REPORT_RETENTION_DAYS} days"
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"Retention purge failed: {e}")


def _preload_models():
    """Preload ML models in background thread."""
    import threading

    def _preload():
        is_lite = (_profile == "lite") or (os.getenv("SLOPTOTAL_LITE") == "1")

        if is_lite:
            # Lite profile: only preload the 4 fast engines used for quick scoring & snippet scans (~250MB total)
            loaders = [
                ("Fakespot RoBERTa", "app.engines.classifier_fakespot", "_init_pool"),
                ("E5-Small LoRA", "app.engines.classifier_e5", "_init_pool"),
                ("TMR RoBERTa", "app.engines.classifier_tmr", "_init_pool"),
                ("BERT-tiny RAID", "app.engines.classifier_bert_raid", "_init_pool"),
            ]
        else:
            loaders = [
                ("GPT-2 Medium", "app.engines.perplexity", "_load_model"),
                ("DistilGPT-2", "app.engines.cross_perplexity", "_load_distil_model"),
                ("OpenAI detector", "app.engines.classifier_openai", "_load_model"),
                ("ChatGPT detector", "app.engines.classifier_chatgpt", "_load_model"),
                ("ReMoDetect DeBERTa", "app.engines.classifier_remodetect", "_load_model"),
                ("Fakespot RoBERTa", "app.engines.classifier_fakespot", "_init_pool"),
                ("E5-Small LoRA", "app.engines.classifier_e5", "_init_pool"),
                ("TMR RoBERTa", "app.engines.classifier_tmr", "_init_pool"),
                ("BERT-tiny RAID", "app.engines.classifier_bert_raid", "_init_pool"),
                ("Desklib DeBERTa", "app.engines.classifier_desklib", "_load_model"),
                (
                    "SuperAnnotate RoBERTa",
                    "app.engines.classifier_superannotate",
                    "_load_model",
                ),
            ]

        import importlib

        for name, module_path, func_name in loaders:
            try:
                mod = importlib.import_module(module_path)
                getattr(mod, func_name)()
                log.info(f"Preloaded {name}")
            except Exception as e:
                log.warning(f"{name} preload failed: {e}")

        log.info(f"Model preloading complete ({len(loaders)} engines ready)")

    threading.Thread(target=_preload, daemon=True).start()


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    db_health = await check_database_health()
    resp = {
        "status": "healthy" if db_health["status"] == "healthy" else "degraded",
        "database": db_health,
        "engines": len(get_engine_list()),
    }
    if app.state.queue_manager:
        resp["queue"] = app.state.queue_manager.queue_status()
    return resp


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
