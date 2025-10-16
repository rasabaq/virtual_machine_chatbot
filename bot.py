import os
import re
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import discord
from discord.ext import commands
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.tools.render import render_text_description
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_react_agent, AgentExecutor

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOKEN_KEY = os.getenv("DISCORD_TOKEN")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

loader1 = TextLoader("memoriatitulo.txt", encoding="utf-8")
loader2 = TextLoader("practicaprofesional.txt", encoding="utf-8")
loader3 = TextLoader("electivos.txt", encoding="utf-8")

docmm = loader1.load()
docpp = loader2.load()
docelec = loader3.load()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_SEARCH_K = 3  # reduce fetched docs per herramienta para respuestas mas rapidas
MEMORY_WINDOW_MESSAGES = 3  # limita historial enviado al modelo manteniendo contexto reciente

emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorMM = FAISS.from_documents(docmm, emb)
vectorPP = FAISS.from_documents(docpp, emb)
vectorEE = FAISS.from_documents(docelec, emb)

retrieverMM = vectorMM.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})
retrieverPP = vectorPP.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})
retrieverEE = vectorEE.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})

llm_thesis = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key= GOOGLE_API_KEY, temperature=0.6)
llm_internship = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key= GOOGLE_API_KEY, temperature=0.6)
llm_electives = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key= GOOGLE_API_KEY, temperature=0.6)

qa_thesis = RetrievalQA.from_chain_type(llm = llm_thesis, retriever = retrieverMM, chain_type = "stuff")
qa_internship = RetrievalQA.from_chain_type(llm = llm_internship, retriever = retrieverPP, chain_type = "stuff")
qa_electives = RetrievalQA.from_chain_type(llm = llm_electives, retriever = retrieverEE, chain_type = "stuff")

tool_thesis = Tool(
    name = "thesis_tool",
    description=("Herramienta para utilizar cuando se realizan preguntas sobre la thesis, las cuales pueden ser sobre plazos, inscripción, temas, requerimientos, etc"),
    func = qa_thesis.run,
)

tool_internship = Tool(
    name = "internship_tool",
    description=("Herramienta para utilizar cuando se realizan preguntas sobre internship, las cuales pueden ser sobre plazos, inscripción, temas, requerimientos, etc"),
    func = qa_internship.run,
)

tool_electives = Tool(
    name= "elective_tool",
    description=("Herramienta para utilizar cuando se realizan preguntas relacionadas con las asignaturas electivas (también conocidas como electivos). "
        "Estas preguntas pueden referirse principalmente a los contenidos contenidos, "),
    func= qa_electives.run,    
)
tools = [tool_thesis, tool_internship, tool_electives]

agent = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)

FORMAT_INSTRUCTIONS = (
    "Sigue este formato ESTRICTO (sin bloques de codigo):\n"
    "Thought: <think>tu razonamiento breve</think>\n"
    "Action: nombre_de_la_herramienta (solo el nombre, sin XML adicional)\n"
    "Action Input: <action_input>{\"query\": \"...\"}</action_input>\n"
    "## tras cada Observation puedes iterar con otro Thought/Action/Action Input ##\n"
    "Final Answer: <final>respuesta final al usuario</final>"
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "<role>Eres un agente ReAct especializado en reglamentos de Memoria de Título, Práctica Profesional y Electivos de la UdeC.</role>\n"
     "<capabilities>"
     "-Si la consulta es small talk (saludos/despedidas), responde breve y cordial SIN usar herramientas.\n"
     "-Usa las herramientas disponibles ({tool_names}) cuando la consulta trate de esos reglamentos\n"
     "-Si falta informacion, dilo sin inventar. \n"
     "</capabilities>"
     "Piensa paso a paso DENTRO de <think>...</think> como tu scratchpad privado.\n"
     "NUNCA muestres ni cites nada dentro de <think>...</think>.\n\n"
     "<policies>"
     "Si la pregunta no se relaciona con estos reglamentos o no puede ser respondida con herramientas, "
     "RESPONDE EXACTAMENTE: "
     "\"Lo siento, no estoy capacitado para responder preguntas fuera del ámbito de la memoria de título, la práctica profesional y electivos.\""
     "</policies>\n"
     "<style>Detallado pero claro, tono cordial. Evita bloques de código para llamadas a herramientas.</style>\n"
     "<char_limit>La respuesta FINAL debe ser ≤ 1800 caracteres (contando espacios) "
     "De ser necesario resume lo mas minimo"
     "Si tu primer borrador excede, resume y prioriza bullets, reglas y pasos clave hasta cumplir el límite.</char_limit>\n"
     "<format>\n{format_instructions}\n</format>\n"
     "<require_tools>Debes responder usando las herramientas disponibles: {tool_names}.</require_tools>"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}"),   
]).partial(
    tools=render_text_description(tools),
    tool_names=", ".join(t.name for t in tools),
    format_instructions=FORMAT_INSTRUCTIONS,  
)


user_executors: dict[int, AgentExecutor] = {}

def make_executor() -> AgentExecutor:
    """Construye un AgentExecutor con memoria vacía."""
    react_agent = create_react_agent(agent, tools, prompt)
    memory = ConversationBufferWindowMemory(
        k=MEMORY_WINDOW_MESSAGES,          # envia solo los turnos mas recientes
        memory_key="chat_history",           # Debe coincidir con MessagesPlaceholder
        return_messages=True,
        input_key="input",
        output_key="output",
    )
    executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        memory=memory,                     # Memoria conectada
        verbose=False,
        handle_parsing_errors=True
    )
    return executor

def extract_final_answer(text: str) -> str:
    """
    Extrae el contenido interno del tag <final>...</final> si existe.
    Se usa para asegurarnos de no mostrar markup al enviar la respuesta.
    """
    if not text:
        return ""
    match = re.search(r"<final>([\s\S]*?)</final>", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def get_executor_for_user(user_id: int) -> AgentExecutor:
    if user_id not in user_executors:
        user_executors[user_id] = make_executor()   # Cada usuario tiene su propio executor+memoria
    return user_executors[user_id]

DB_PATH = Path(__file__).resolve().parent / "chatlog.sqlite3"

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            username TEXT,
            channel_id TEXT,
            channel_name TEXT,
            question TEXT,
            answer TEXT,
            created_at TEXT
        );
        """
    )
    conn.commit()
    conn.close()

def log_interaction(*, user_id, username, channel_id, channel_name, question, answer) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO interactions (user_id, username, channel_id, channel_name, question, answer, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(user_id),
            username or "",
            str(channel_id),
            channel_name or "",
            question or "",
            answer or "",
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
        ),
    )
    conn.commit()
    conn.close()

@bot.event
async def on_ready():
    init_db()
    logger.info(f"✅ Bot conectado como {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    try:
        user_text = message.content.strip()
        if not user_text:
            return

        if not tools:
            await message.channel.send(
                f"{message.author.mention} No tengo fuentes cargadas aún. "
                "Sube o configura los documentos necesarios. ¿En qué puedo ayudarte?"
            )
            await bot.process_commands(message)
            return

        # Executor con memoria propia para ESTE usuario
        executor = get_executor_for_user(message.author.id)

        # Llamada asíncrona al agente
        res = await executor.ainvoke({"input": user_text})
        raw_output = (res.get("output", "") or "").strip()
        output = extract_final_answer(raw_output) or "No pude generar una respuesta."

        try:
            log_interaction(
                user_id=message.author.id,
                username=str(message.author),
                channel_id=message.channel.id,
                channel_name=getattr(message.channel, "name", "DM"),
                question=user_text,
                answer=output,
            )
        except Exception:
            logger.exception("No se pudo guardar la interacción en SQLite")

        # Etiqueta SIEMPRE a la persona
        await message.channel.send(f"{message.author.mention} 📘 Respuesta:\n{output}")

    except Exception as e:
        logger.exception("Error procesando el mensaje")
        await message.channel.send(f"{message.author.mention} ❌ Error: {e}")

    # Mantén compatibilidad con prefijo !
    await bot.process_commands(message)

@bot.command(name="ultimas")
async def ultimas(ctx, n: int = 10):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT username, question, created_at
            FROM interactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (n,),
        )
        rows = cur.fetchall()
        conn.close()

        if not rows:
            await ctx.send("No hay registros aún.")
            return

        lines = []
        for uname, q, ts in rows:
            snippet = (q or "").replace("\n", " ")
            if len(snippet) > 180:
                snippet = snippet[:180] + "…"
            lines.append(f"• **{uname}** [{ts}]: {snippet}")

        await ctx.send(f"**Últimas {len(rows)} preguntas:**\n" + "\n".join(lines))

    except Exception as e:
        logger.exception("Error consultando SQLite")
        await ctx.send(f"¡Error consultando DB: {e}!")

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    bot.run(TOKEN_KEY)
    



