from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from src.logger import logging
from src.utils.helpers import read_template

router = APIRouter(tags=["Pages"])


def render_template_safely(template_name: str) -> str:
    try:
        return read_template(template_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template {template_name} not found")
    except Exception as exc:
        logging.exception("Failed to render %s template", template_name)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@router.get("/", response_class=HTMLResponse)
@router.get("/landing", response_class=HTMLResponse)
async def landing_page():
    return render_template_safely("index.html")


@router.get("/home", response_class=HTMLResponse)
@router.get("/app", response_class=HTMLResponse)
async def chat_app():
    return render_template_safely("home.html")


@router.get("/login", response_class=HTMLResponse)
async def login_page():
    return render_template_safely("login.html")
