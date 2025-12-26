from fastapi import FastAPI, Request, Form, Query, Body
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from app.core.services import graph_optimized_search, get_suggestions, get_paper_details
from app.core.gemini_service import analyze_paper_content, chat_with_paper_context
from app.core.config import settings
from app.core.graph import graph_db
import uvicorn
import markdown

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Mount Static Files (CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup Templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/suggest")
def suggest(q: str = Query(..., min_length=1)):
    data = get_suggestions(q)
    return JSONResponse(content=data)

@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = Query(..., min_length=3), page: int = Query(1, ge=1)):
    # Perform the graph-optimized search
    search_data = await graph_optimized_search(q, page=page, limit=15)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "query": q,
        "results": search_data["results"],
        "latency": search_data["latency"],
        "routed_to": search_data["routed_to"],
        "page": page,
        "has_next": len(search_data["results"]) >= 15 # Simple check for next page
    })

@app.get("/analysis/{collection}/{paper_id}", response_class=HTMLResponse)
async def analysis(request: Request, collection: str, paper_id: str):
    # 1. Fetch Paper Details
    paper = get_paper_details(collection, paper_id)
    
    if not paper:
        return HTMLResponse("Paper not found", status_code=404)
    
    # Render page immediately with paper details, analysis will be fetched via JS
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "paper": paper
    })

@app.get("/api/analyze/{collection}/{paper_id}")
async def analyze_paper_api(collection: str, paper_id: str):
    paper = get_paper_details(collection, paper_id)
    if not paper:
        return JSONResponse({"error": "Paper not found"}, status_code=404)
        
    abstract_text = paper.get("abstract") or "Abstract not available."
    analysis_md = await analyze_paper_content(paper["title"], abstract_text)
    
    # Convert Markdown to HTML
    analysis_html = markdown.markdown(analysis_md)
    
    return JSONResponse({"analysis": analysis_html})

@app.post("/chat")
async def chat(
    message: str = Body(...), 
    history: list = Body([]), 
    context: str = Body(...)
):
    response = await chat_with_paper_context(history, message, context)
    return JSONResponse({"response": response})

@app.post("/feedback")
async def feedback(collection: str = Body(...), reward: float = Body(0.1)):
    """
    Updates the graph weights based on user interaction (e.g., clicking a result).
    """
    graph_db.update(collection, reward)
    return JSONResponse({"status": "success", "new_weight": graph_db.get_weight(collection)})

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
