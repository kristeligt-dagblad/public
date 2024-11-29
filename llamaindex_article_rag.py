"""
title: LlamaIndex + Gemini Test Pipeline
author: analyse@k.dk
date: 2024-11-15
version: 1.0
description: Test pipeline for verifying LlamaIndex and Gemini functionality
requirements: llama-index, llama-index-llms-gemini
"""

from typing import Generator, Iterator, List, Union
from pydantic import BaseModel
from llama_index.llms.gemini import Gemini
from llama_index.core import Document, VectorStoreIndex, Settings

class Pipeline:
    """Test Pipeline for Llama Index functionality"""
    
    class Valves(BaseModel):
        """Test options for WebUI"""
        test_text: str = "The capital of France is Paris. The capital of Italy is Rome."

    def __init__(self):
        self.documents = None
        self.index = None
        self.valves = self.Valves()

    async def on_startup(self):
        try:
            # Create a test document
            test_doc = Document(text=self.valves.test_text)
            
            # Initialize settings with default embedding model
            Settings.llm = Gemini(model="gemini-1.5-pro-001")
            
            # Create index from test document
            self.index = VectorStoreIndex.from_documents([test_doc])
            print("Llama Index initialization successful!")
            return True
            
        except Exception as e:
            print(f"Llama Index initialization failed: {str(e)}")
            return False

    async def on_shutdown(self):
        print("Pipeline shutdown")

    async def on_valves_updated(self) -> None:
        print(f"Updating index with new test text: {self.valves.test_text}")
        await self.on_startup()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            # Test querying the index
            if self.index is None:
                return "Index not initialized. Please check startup logs."
                
            query_engine = self.index.as_query_engine(streaming=True)
            response = query_engine.query(user_message)
            
            return response.response_gen
            
        except Exception as e:
            return f"Query failed: {str(e)}"
