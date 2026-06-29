SYSTEM_PROMPT = """You are an expert aviation instructor. Answer questions based ONLY on the provided context. If the context does not contain the answer, say so clearly. Always cite sources in [brackets]."""

RAG_PROMPT_TEMPLATE = """
Context:
{context}

---
Question: {question}

Answer (with source citations in [brackets]):"""

MULTI_TURN_PROMPT_TEMPLATE = """
Previous conversation:
{history}

Context:
{context}

---
Follow-up question: {question}

Answer (with source citations in [brackets]):"""

REWRITE_PROMPT_TEMPLATE = """You are a search query optimizer for an aviation knowledge base. Rewrite the user's question into a keyword-rich search query optimized for retrieval.

Rules:
- Use technical, formal aviation terminology
- Expand abbreviations (V1 -> V1决策速度, APU -> 辅助动力装置)
- Resolve pronouns based on conversation history if provided
- Keep key Chinese/English terms in original language
- Output ONLY the rewritten query, no explanation

Conversation history:
{history}

User question: {query}
Rewritten query:"""

ROUTE_PROMPT_TEMPLATE = """Classify this aviation question into exactly one category:
- aircraft_structure: questions about aircraft parts, systems, components, design, structure
- flight_ops: questions about flying procedures, cockpit operations, checklists, maneuvers
- regulations: questions about aviation laws, rules, certification requirements, safety regulations

Question: {query}
Category:"""
