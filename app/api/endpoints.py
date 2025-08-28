from fastapi import APIRouter 
from pydantic import BaseModel
from app.services.text_kis_service import TextKISService
from app.services.qa_service import QAService 
from app.services.trake_service import TrakeService 

router = APIRouter()
kis_service = TextKISService()
qa_service = QAService()
trake_service = TrakeService()

class QueryRequest(BaseModel):
    query: str 
    k: int = 5 

@router.post("/kis")
def kis_search(request: QueryRequest):
    return kis_service.search(request.query, request.k)

@router.post("/qa")
def qa_search(request: QueryRequest):
    return qa_service.search(request.query, request.k)

@router.post("/trake")
def trake_search(request: QueryRequest):
    return trake_service.search(request.query, request.k)