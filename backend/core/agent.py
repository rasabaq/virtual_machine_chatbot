import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .rag import rag_system

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un asistente virtual especializado en reglamentos de la Facultad de Ingeniería de la UdeC:
- Memoria de Título
- Práctica Profesional
- Electivos
- Reglamento Interno de Docencia de Pregrado (inscripción de asignaturas, calificaciones, continuación de estudios, cambios de carrera, convalidaciones, suspensión/reincorporación, graduación y titulación)

INSTRUCCIONES:
1. Si el usuario saluda o hace small talk, responde brevemente y cordialmente.
2. Si pregunta sobre los temas anteriores, usa el CONTEXTO proporcionado para responder.
3. Si la pregunta no está en tu ámbito, responde: "Lo siento, no estoy capacitado para responder preguntas fuera del ámbito de la memoria de título, la práctica profesional, electivos y el reglamento interno de docencia."
4. Sé detallado pero claro, con tono cordial.
5. Máximo 1800 caracteres en tu respuesta.

CONTEXTO RELEVANTE:
{context}
"""

CLASSIFIER_PROMPT = """Clasifica la siguiente pregunta en una de estas categorías:
- thesis: si es sobre memoria de título
- internship: si es sobre práctica profesional
- electives: si es sobre electivos/asignaturas electivas
- regulations: si es sobre reglamento interno de docencia, inscripción de asignaturas, calificaciones, aprobación, créditos, baja académica, continuación de estudios, cambio de carrera, ingreso especial, convalidación, revalidación, suspensión, renuncia, reincorporación, graduación o titulación
- none: si es saludo, small talk, o tema no relacionado

Pregunta: {question}

Responde SOLO con una palabra: thesis, internship, electives, regulations, o none."""


class SimpleAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.llm = ChatOpenAI(
            model="gpt-5-nano",
            api_key=api_key,
        )
        
        self.classifier_chain = (
            ChatPromptTemplate.from_template(CLASSIFIER_PROMPT)
            | self.llm
            | StrOutputParser()
        )
        
        self.response_chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{question}")
            ])
            | self.llm
            | StrOutputParser()
        )
    
    async def invoke(self, question: str) -> str:
        # Step 1: Classify the question (1 API call)
        try:
            category = await self.classifier_chain.ainvoke({"question": question})
            category = category.strip().lower()
            logger.info(f"Question classified as: {category}")
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            category = "none"
        
        # Step 2: Get RAG context if needed (uses cached embeddings)
        context = ""
        if category in ["thesis", "internship", "electives", "regulations"]:
            try:
                context = rag_system.query(category, question)
            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                context = "No se pudo obtener información relevante."
        else:
            context = "No se requiere contexto específico para esta consulta."
        
        # Step 3: Generate response (1 API call)
        try:
            response = await self.response_chain.ainvoke({
                "context": context,
                "question": question
            })
            return response
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Error al generar respuesta: {e}"


def create_agent_executor():
    """Factory function to maintain compatibility with existing code."""
    return SimpleAgent()
