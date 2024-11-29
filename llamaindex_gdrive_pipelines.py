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

class Pipeline:
    """Test Pipeline for LlamaIndex and Gemini"""
    
    class Valves(BaseModel):
        """Test configuration"""
        test_query: str = "What is AI?"

    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):
        from llama_index.core import Settings, VectorStoreIndex, Document
        from llama_index.llms.gemini import Gemini

        try:
            # Set up Gemini
            Settings.llm = Gemini(model="gemini-1.5-pro-001")
            
            # Create test document
            test_text = (
                "Artificial Intelligence (AI) is the simulation of human intelligence by machines. "
                "Machine learning is a subset of AI that enables systems to learn from data. "
                "Deep learning is a type of machine learning based on neural networks."
            )
            self.documents = [Document(text=test_text)]
            
            # Create index
            self.index = VectorStoreIndex.from_documents(self.documents)
            
            print("✓ LlamaIndex and Gemini setup successful")
            print("✓ Test document indexed")
            
        except Exception as e:
            print(f"❌ Setup failed: {str(e)}")
            raise

    async def on_shutdown(self):
        print("Pipeline shutdown")

    async def on_valves_updated(self) -> None:
        print(f"Testing with query: {self.valves.test_query}")
        await self.on_startup()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            # Create query engine
            query_engine = self.index.as_query_engine(streaming=True)
            
            # Generate response
            response = query_engine.query(user_message)
            
            print(f"✓ Query processed: {user_message}")
            return response.response_gen
            
        except Exception as e:
            error_msg = f"❌ Query failed: {str(e)}"
            print(error_msg)
            
            def error_generator():
                yield error_msg
                
            return error_generator()
