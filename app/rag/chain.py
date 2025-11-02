"""
RAG Chain implementation using LangChain
Combines retrieval and generation for question answering
"""
from typing import Dict, Any, List
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from app.rag.retriever import PineconeRetriever

logger = logging.getLogger(__name__)


class RAGChain:
    """RAG chain that retrieves context and generates answers"""

    def __init__(
        self,
        retriever: PineconeRetriever,
        openai_api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the RAG chain

        Args:
            retriever: Document retriever instance
            openai_api_key: OpenAI API key
            model_name: Name of the OpenAI model
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.retriever = retriever

        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Define prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful financial assistant that answers questions based on provided context.

Your responsibilities:
1. Answer questions accurately using ONLY the information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite the document sources when relevant
4. Provide clear, concise, and professional responses
5. Focus on financial accuracy and clarity

Context:
{context}"""),
            ("human", "{question}")
        ])

        # Create chain
        self.chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        logger.info("RAG chain initialized successfully")

    def invoke(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Process a question through the RAG pipeline

        Args:
            question: User's question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing question: {question[:50]}...")

        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

        if not retrieved_docs:
            logger.warning("No documents retrieved")
            return {
                "question": question,
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "retrieved_docs": []
            }

        # Format context
        context = self.retriever.format_docs_for_context(retrieved_docs)

        # Generate answer
        logger.info("Generating answer...")
        answer = self.chain.invoke({
            "context": context,
            "question": question
        })

        # Extract sources
        sources = list(set([doc.get("source", "unknown") for doc in retrieved_docs]))

        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_docs": [
                {
                    "text": doc["text"][:200] + "...",  # Truncate for response
                    "source": doc["source"],
                    "score": doc["score"]
                }
                for doc in retrieved_docs
            ]
        }

        logger.info("Answer generated successfully")
        return result

    def invoke_with_chat_history(
        self,
        question: str,
        chat_history: List[Dict[str, str]],
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Process a question with chat history for context

        Args:
            question: User's question
            chat_history: List of previous Q&A pairs
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with answer and metadata
        """
        # Reformulate question based on chat history if needed
        if chat_history:
            history_context = "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}"
                for item in chat_history[-3:]  # Last 3 exchanges
            ])
            contextualized_question = f"Previous conversation:\n{history_context}\n\nCurrent question: {question}"
        else:
            contextualized_question = question

        # Use the standard invoke method
        return self.invoke(contextualized_question, top_k=top_k)
