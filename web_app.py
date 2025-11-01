import os
import secrets
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_core import query_rag


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTICLE_ROOT = APP_ROOT / "article_splits"
ARTICLE_ROOT = Path(os.getenv("RAG_ARTICLE_ROOT", str(DEFAULT_ARTICLE_ROOT))).resolve()
PDF_AVAILABLE = ARTICLE_ROOT.exists()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 6

app = FastAPI(title="물류법 RAG 검색기")
templates = Jinja2Templates(directory=str(APP_ROOT / "templates"))

if PDF_AVAILABLE:
    app.mount("/pdf", StaticFiles(directory=str(ARTICLE_ROOT)), name="pdf")


class ConversationTurn(dict):
    """Helper structure stored per conversation turn."""

    @property
    def history_tuple(self):
        return self["user"], self["assistant"]

    @property
    def user(self) -> str:
        return self.get("user", "")

    @property
    def assistant(self) -> str:
        return self.get("assistant", "")

    @property
    def contexts(self) -> List[dict]:
        return self.get("contexts", [])


sessions: Dict[str, List[ConversationTurn]] = {}


def _get_session_id(request: Request) -> str:
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = secrets.token_hex(16)
    return session_id


def _trim_context(ctx: dict) -> dict:
    text = ctx.get("text", "")
    snippet = text.replace("\n", " ")[:240]
    rel_path = ctx.get("file", "")
    rel_posix = Path(rel_path).as_posix()
    law_name = ctx.get("law_name", rel_path.split("/", 1)[0] if rel_path else "")
    return {
        "file": rel_path,
        "law_name": law_name,
        "pages": f"p.{ctx['page_start']}-{ctx['page_end']}",
        "score": round(ctx.get("score", 0.0), 3),
        "snippet": snippet,
        "url_path": rel_posix,
    }


def _history_for_api(turns: List[ConversationTurn]):
    return [turn.history_tuple for turn in turns]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session_id = _get_session_id(request)
    turns = sessions.setdefault(session_id, [])
    response = templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "turns": turns,
            "config": {
                "embed_model": EMBED_MODEL,
                "chat_model": CHAT_MODEL,
                "top_k": TOP_K,
            },
            "pdf_available": PDF_AVAILABLE,
            "last_query": "",
        },
    )
    response.set_cookie("session_id", session_id, httponly=True, max_age=7 * 24 * 3600)
    return response


@app.post("/query", response_class=HTMLResponse)
async def handle_query(request: Request, query: str = Form("")):
    session_id = _get_session_id(request)
    turns = sessions.setdefault(session_id, [])

    if not query.strip():
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "turns": turns,
                "error": "검색 질의를 입력해 주세요.",
                "config": {
                    "embed_model": EMBED_MODEL,
                    "chat_model": CHAT_MODEL,
                    "top_k": TOP_K,
                },
                "pdf_available": PDF_AVAILABLE,
                "last_query": query,
            },
        )

    history_pairs = _history_for_api(turns)
    answer, contexts = query_rag(
        index_root=str(ARTICLE_ROOT),
        query=query,
        embed_model=EMBED_MODEL,
        chat_model=CHAT_MODEL,
        top_k=TOP_K,
        force_rebuild=False,
        history_turns=history_pairs,
    )

    trimmed_contexts = [_trim_context(ctx) for ctx in contexts[:TOP_K]]
    turn = ConversationTurn(user=query, assistant=answer, contexts=trimmed_contexts)
    turns.append(turn)

    response = templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "turns": turns,
            "config": {
                "embed_model": EMBED_MODEL,
                "chat_model": CHAT_MODEL,
                "top_k": TOP_K,
            },
            "pdf_available": PDF_AVAILABLE,
            "last_query": "",
        },
    )
    response.set_cookie("session_id", session_id, httponly=True, max_age=7 * 24 * 3600)
    return response


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("RAG_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("RAG_WEB_PORT", "8000"))
    reload_flag = os.getenv("RAG_WEB_RELOAD", "true").lower() in {"1", "true", "yes"}

    uvicorn.run("web_app:app", host=host, port=port, reload=reload_flag)

