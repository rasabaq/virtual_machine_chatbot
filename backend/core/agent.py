import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .rag import rag_system

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un asistente virtual especializado en reglamentos de la Facultad de Ingeniería de la UdeC:
- Memoria de Título
- Práctica Profesional
- Electivos
- Reglamento Interno de Docencia de Pregrado

INSTRUCCIONES:
1. Si el usuario saluda o hace small talk, responde brevemente (1-2 oraciones).
2. Si pregunta sobre los temas anteriores, usa el CONTEXTO para dar una respuesta concisa y directa.
3. Si la información no está en el CONTEXTO, responde: "No encontré esa información en los reglamentos disponibles."
4. Usa viñetas cuando sea apropiado para mayor claridad.

RESTRICCIONES:
- NUNCA ofrezcas redactar documentos, borradores o correos.
- Tu rol es INFORMAR sobre reglamentos, NO ejecutar tareas.
- No pidas datos personales.

CONTEXTO RELEVANTE:
{context}
"""

# El clasificador ahora maneja preguntas mixtas mejor
CLASSIFIER_PROMPT = """Clasifica la siguiente pregunta. Si toca VARIOS temas, elige el más relevante.

Categorías:
- thesis: memoria de título, proyecto de título, informe de memoria
- internship: práctica profesional, práctica industrial
- electives: electivos, asignaturas optativas, ramos optativos
- regulations: inscripción, calificaciones, créditos, baja académica, convalidación, 
               graduación, titulación, reglamento de docencia, suspensión, reincorporación
- none: saludo, small talk, temas no relacionados con los reglamentos

Pregunta: {question}

Responde SOLO con una palabra."""

class SimpleAgent:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
        )

        self.classifier_chain = (
            ChatPromptTemplate.from_template(CLASSIFIER_PROMPT)
            | self.llm
            | StrOutputParser()
        )

    async def invoke(self, question: str, history: list = []) -> str:
        logger.info(f"[AGENT] Processing: {question[:80]}")

        # Paso 1: Clasificar
        try:
            category = await self.classifier_chain.ainvoke({"question": question})
            category = category.strip().lower()
            # Sanity check por si el modelo devuelve texto extra
            valid = {"thesis", "internship", "electives", "regulations", "none"}
            if category not in valid:
                category = "none"
            logger.info(f"[AGENT] Category: {category}")
        except Exception as e:
            logger.error(f"[AGENT] Classification failed: {e}")
            category = "none"

        # Paso 2: Obtener contexto RAG
        context = "No se requiere contexto para esta consulta."
        if category != "none":
            try:
                context = rag_system.query(category, question)
                logger.info(f"[AGENT] RAG context: {len(context)} chars")
            except Exception as e:
                logger.error(f"[AGENT] RAG failed: {e}")
                context = "No se pudo obtener información relevante."

        # Paso 3: Construir mensajes con historial
        # El system prompt va primero con el contexto ya formateado
        messages = [
            ("system", SYSTEM_PROMPT.format(context=context))
        ]

        # Agregar historial previo (ya viene sin el mensaje actual gracias al fix en main.py)
        for msg in history:
            role = "human" if msg["role"] == "user" else "assistant"
            messages.append((role, msg["content"]))

        # Agregar pregunta actual
        messages.append(("human", question))

        # Paso 4: Generar respuesta con chain local (no sobreescribe self)
        try:
            response_chain = (
                ChatPromptTemplate.from_messages(messages)
                | self.llm
                | StrOutputParser()
            )
            # No necesita variables porque el system ya está formateado
            response = await response_chain.ainvoke({})
            logger.info(f"[AGENT] Response: {len(response)} chars")
            return response
        except Exception as e:
            logger.error(f"[AGENT] Response failed: {e}")
            return f"Error al generar respuesta: {e}"



def create_agent_executor():
    """Factory function to maintain compatibility with existing code."""
    return SimpleAgent()