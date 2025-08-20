import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
from typing import Dict, Any, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from pydantic import BaseModel
import xml.etree.ElementTree as ET
import logging
from langchain.chains import RetrievalQA
import discord
from discord.ext import commands

load_dotenv()  
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOKEN_KEY = os.getenv("DISCORD_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#inicializar el bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


class ClassificationResult(BaseModel):
    """Modelo para el resultado de clasificación"""
    topic: Literal["thesis", "internship", "conversation"]
    confidence: float
    reasoning: str

class XMLClassificationParser(BaseOutputParser):
    """Parser personalizado para extraer clasificación desde XML"""  
    def parse(self, text: str) -> ClassificationResult:
        try:
            # Limpiar el texto y extraer solo el XML
            xml_start = text.find('<classification>')
            xml_end = text.find('</classification>') + len('</classification>')
            
            if xml_start == -1 or xml_end == -1:
                raise ValueError("No se encontró estructura XML válida")
            
            xml_content = text[xml_start:xml_end]
            root = ET.fromstring(xml_content)
            
            topic = root.find('topic').text
            confidence = float(root.find('confidence').text)
            reasoning = root.find('reasoning').text
            
            return ClassificationResult(
                topic=topic,
                confidence=confidence,
                reasoning=reasoning
            )
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            # Fallback a TOPIC_A en caso de error
            return ClassificationResult(
                topic="thesis",
                confidence=0.5,
                reasoning="Error en clasificación, usando fallback"
            )
loader1 = TextLoader("memoriatitulo.txt", encoding="utf-8")
loader2 = TextLoader("practicaprofesional.txt", encoding="utf-8")
docmm = loader1.load()
docpp = loader2.load()
embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorstoreMM = FAISS.from_documents(docmm, embedding)
vectorstorePP = FAISS.from_documents(docpp, embedding)   

class MultiModelSystem:
    """Sistema principal multimodelo con LangChain"""
    
    def __init__(self, vectorstoreMM, vectorstorePP):
        self.mm_vectorstore = vectorstoreMM
        self.mp_vectorstore = vectorstorePP
        # Crear todos los modelos con Gemini
        self.classifier_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1, # Baja temperatura para clasificación precisa
            max_tokens=500 
        )
        
        self.mm_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            max_tokens=500 
        )
        
        self.mp_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7, 
            max_tokens=500 
        )
        
        # Configurar parser
        self.classification_parser = XMLClassificationParser()
        self._setup_chains()
    
    def _setup_chains(self):
        """Configurar las chains de LangChain"""
        
        # Prompt para clasificación
        classification_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
Eres un clasificador experto de la Universidad de Concepción que debe identificar el tipo de pregunta.

INSTRUCCIONES:
- Analiza la pregunta del usuario
- Clasifica en uno de estos tres topics:
  * thesis: Preguntas relacionadas a inscripción, evaluación o detalles sobre la memoria de titulo o thesis. Considera que suele llamarse como MT, thesis, memoria, proyecto de titulo. No tiene que estar explicitamente escrito memoria de titulo
  * internship: Preguntas sobre cuando hacer la practica profesional, requisitos, seguro o necesidades. A veces le llaman PP, o usan abreviaturas como practica prof., prac profe. No tiene que estar explicito escrito practica profesional o internship
  * conversation: Preguntas relacionadas a saludos, cómo estas, despedidas o conversaciones basicas humanas simple.

- Proporciona tu respuesta SOLO en formato XML
- Si el tema preguntado no se encuentra en alguna de las clasificaciones, responde que aquello no esta en tu capacidad de respuestas.
- Si el tema preguntado es una conversation, responde de manera coordial y positiva.

PREGUNTA: {question}


RESPUESTA (formato XML obligatorio):
<classification>
    <topic>thesis, internship o conversation</topic>
    <confidence>0.0 a 1.0</confidence>
    <reasoning>Explicación breve de por qué elegiste este topic</reasoning>
</classification>

"""
        )
        conversation_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
        Eres un asistente creado por alumnos de Ingeniería civil Industrial con fines de ayudar a todo los estudiantes de la carrera, además 
        dentro de tus hobbys esta ir a ver la maravillosas clases que imparte el profesor Jorge Jimenez y Carlos Contreras
        
        Intrucciones:
        - Responde de manera cordial y clara solo preguntas relacionadas a quién eres, cómo estas, saludos o despedidas
        - Si la pregunta se escapa de los puntos previos, no respondas o inventes.
        - Siempre termina tu respuesta preguntan en que puedes ayudar.

        Responde de manera clara y corta la siguiente pregunta:
        {question}
        """
        )

        
        mm_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
Eres un asistente especializado en información sobre la Memoria de Título de Ingeniería Civil Industrial en la Universidad de Concepción.

Intrucciones: 
- Analiza la pregunta cuidadosamente
- Desarrolla pensamiento paso a paso
- Responde solo preguntas que este relacionadas al topico en el que eres experto


Responde de manera clara y breve la siguiente pregunta:
{question}

"""
        )
        
        # Prompt para Practica Profesional
        mp_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
Eres un asistente especializado en información sobre la practica profesional de Ingeniería Civil Industrial en la Universidad de Concepción.

Instrucciones:
- Analiza la pregunta cuidadosa mente
- Desarrolla pensamiento paso a paso
- Responde solo preguntas que este relacionadas al topico en el que eres experto

Responde de manera clara y de manera breve la siguiente pregunta:
{question}

"""
        )
        # Crear chains
        self.classification_chain = LLMChain(
            llm=self.classifier_model,
            prompt=classification_prompt,
            output_parser=self.classification_parser
        )
        self.conversations = LLMChain(
            llm = self.classifier_model,
            prompt=conversation_prompt
        )
        
        # RetrievalQA para Memoria de Título
        self.mm_qa = RetrievalQA.from_chain_type(
            llm=self.mm_model,
            retriever=self.mm_vectorstore.as_retriever(),
            chain_type="stuff"  # Puedes cambiar a refine/map_reduce si prefieres
        )

        # RetrievalQA para Práctica Profesional
        self.mp_qa = RetrievalQA.from_chain_type(
            llm=self.mp_model,
            retriever=self.mp_vectorstore.as_retriever(),
            chain_type="stuff"
        )

    def process_question(self, question: str) -> Dict[str, Any]:
        logger.info(f"Procesando pregunta: {question}")
        classification_result = self.classification_chain.run(question=question)

        logger.info(f"Clasificación: {classification_result.topic} (confianza: {classification_result.confidence})")

        # Enrutamiento
        if classification_result.topic == "thesis":
            specialized_response = self.mm_qa.run(question)
            model_used = ""
        elif classification_result.topic == "internship":
            specialized_response = self.mp_qa.run(question)
            model_used = "internship model"
        elif classification_result.topic == "conversation":
            specialized_response = self.conversations.run(question)
            model_used = "conversation model"
        else:
            specialized_response = "Lo siento, esta pregunta no está en mi rango de respuestas."
            model_used = "Ninguno (clasificación fuera de dominio)"

        return {
            "question": question,
            "classification": {
                "topic": classification_result.topic,
                "confidence": classification_result.confidence,
                "reasoning": classification_result.reasoning
            },
            "model_used": model_used,
            "response": specialized_response,
            "workflow_steps": [
                "1. Clasificación con Gemini",
                f"2. Enrutamiento a {model_used}",
                "3. Recuperación y respuesta con contexto vía VectorStore"
            ]
        }


system = MultiModelSystem(vectorstorePP=vectorstorePP,vectorstoreMM=vectorstoreMM)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # Evita que el bot se responda a sí mismo

    try:
        result = system.process_question(message.content)
        response = result['response']
        await message.channel.send(f"📘 Respuesta:\n{response}")
    except Exception as e:
        await message.channel.send(f"❌ Error: {str(e)}")

if __name__ == "__main__":

    bot.run(TOKEN_KEY)


