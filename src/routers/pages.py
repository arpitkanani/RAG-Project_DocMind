
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from src.logger import logging
from src.utils.helpers import read_template

router = APIRouter(tags=["Pages"])


def render_chat_template() -> str:
    try:
        return read_template("home.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend template not found")
    except Exception as exc:
        logging.exception("Failed to render app template")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Browsers request this automatically on every page load. We don't ship
    # an icon file, so just return an empty 204 instead of a 404 -- purely
    # cosmetic, keeps the console/log clean.
    return Response(status_code=204)


@router.get("/", response_class=HTMLResponse)
@router.get("/app", response_class=HTMLResponse)
@router.get("/home", response_class=HTMLResponse)
async def chat_app():
    return render_chat_template()


@router.get("/landing", response_class=HTMLResponse)
async def landing_page():
    try:
        return read_template("index.html")
    except FileNotFoundError:
        return """
        <html><body>
            <h1>DocMind API Running</h1>
            <p>Visit <a href="/docs">/docs</a> for API documentation</p>
        </body></html>
        """
    except Exception as exc:
        logging.exception("Failed to render landing page")
        raise HTTPException(status_code=500, detail=str(exc))
