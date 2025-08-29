# --------------------------------------------------------------------------
# File: app/chains.py
# Description: Defines the LangChain Expression Language (LCEL) chain for the RAG system.
# --------------------------------------------------------------------------
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.chat_models import ChatOllama

from app.retriever import get_retriever
from app.config import settings

# --- Prompt Template with Guardrails ---
PROMPT_TEMPLATE = """
**Safety Guardrail:** Your primary task is to answer the user's question based *only* on the provided context.
Do not use any external knowledge or information you were trained on.

**Instructions:**
1.  Read the user's question and the context carefully.
2.  If the context contains enough information to answer the question, formulate a concise answer.
3.  Cite the source document's filename for each piece of information you use. For example: "The sky is blue [source: science_paper.pdf]."
4.  If the context does not contain information to answer the question, you MUST respond with exactly: "I don't have enough information to answer that question."
5.  Refuse to answer any questions that are harmful, unethical, or off-topic.

**Context:**
{context}

**Question:**
{question}

**Answer:**
"""

def format_docs(docs):
    """Helper function to format retrieved documents into a single string for the prompt."""
    return "\n\n".join(
        f"Source Filename: {doc.metadata.get('filename', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )

def get_full_chain(retriever=None):
    """
    Creates and returns the main RAG chain that provides both the answer and the source documents.
    This function now accepts a retriever instance to allow for dynamic mode selection.

    Args:
        retriever: An initialized retriever instance. If None, a default one is created.
    
    Returns:
        A LangChain runnable that returns a dictionary with 'answer' and 'context' keys.
    """
    if retriever is None:
        # If no retriever is provided, create a default one (hybrid mode).
        retriever = get_retriever()

    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # This chain takes a question, retrieves documents, and formats them.
    retrieval_and_formatting_chain = (
        RunnablePassthrough.assign(context=itemgetter("question") | retriever)
    )

    # This chain generates the answer using the formatted context and question.
    answer_generation_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    # The final chain combines the two, ensuring the original context is preserved for citations.
    chain = retrieval_and_formatting_chain | {
        "answer": answer_generation_chain,
        "context": itemgetter("context")
    }
    
    return chain

