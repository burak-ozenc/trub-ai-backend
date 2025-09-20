# from fastapi import FastAPI
# import ollama
# from pydantic import BaseModel
# 
# app = FastAPI()  # This line must be at module level, not inside a function
# 
# class GenerateRequest(BaseModel):
#     model: str
#     prompt: str
# 
# @app.post("/api/generate")
# async def generate(request: GenerateRequest):
#     response = ollama.generate(
#         model=request.model,
#         prompt=request.prompt
#     )
#     return {"response": response["response"]}