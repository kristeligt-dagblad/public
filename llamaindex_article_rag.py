"""
title: LlamaIndex + Gemini Test Pipeline (Fixed)
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
        # Initialize immediately instead of waiting for async startup
        self._initialize_index()

    def _initialize_index(self):
        """Synchronous initialization of the index"""
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

    async def on_startup(self):
        # Startup is now just a check since initialization happens in __init__
        if self.index is None:
            self._initialize_index()
        print("✓ Pipeline ready")

    async def on_shutdown(self):
        print("Pipeline shutdown")

    async def on_valves_updated(self) -> None:
        print(f"Testing with query: {self.valves.test_query}")
        if self.index is None:
            self._initialize_index()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            if self.index is None:
                raise ValueError("Index not initialized. Attempting to reinitialize...")
                
            # Create query engine
            query_engine = self.index.as_query_engine(streaming=True)
            
            # Generate response
            response = query_engine.query(user_message)
            
            print(f"✓ Query processed: {user_message}")
            return response.response_gen
            
        except Exception as e:
            error_msg = f"❌ Query failed: {str(e)}"
            print(error_msg)
            
            # Try to recover by reinitializing
            try:
                self._initialize_index()
                query_engine = self.index.as_query_engine(streaming=True)
                response = query_engine.query(user_message)
                print("✓ Recovery successful")
                return response.response_gen
            except Exception as recovery_e:
                def error_generator():
                    yield f"❌ Recovery failed: {str(recovery_e)}"
                return error_generator()