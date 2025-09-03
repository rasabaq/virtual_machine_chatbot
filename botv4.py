import os, logging
from dotenv import load_dotenv
load_dotenv()

import discord
from discord.ext import commands
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.tools.render import render_text_description

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
docmm = loader1.load()
docpp = loader2.load()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vectorMM = FAISS.from_documents(docmm, emb)
vectorPP = FAISS.from_documents(docpp, emb)

llm_thesis = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key= GOOGLE_API_KEY, temperature=0.6)
llm_internship = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key= GOOGLE_API_KEY, temperature=0.6)

qa_thesis = RetrievalQA.from_chain_type(llm = llm_thesis, retriever = vectorMM.as_retriever(), chain_type = "stuff")
qa_internship = RetrievalQA.from_chain_type(llm = llm_internship, retriever = vectorPP.as_retriever(), chain_type = "stuff")

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

tools = [tool_thesis, tool_internship]

agent = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)

FORMAT_INSTRUCTIONS = (
    "Sigue este formato ESTRICTO (sin bloques de código):\n"
    "Thought: tu razonamiento breve\n"
    "Action: <nombre_de_la_herramienta>\n"
    "Action Input: {\"query\": \"...\"}\n"
    "## tras cada Observation puedes iterar con otro Thought/Action/Action Input ##\n"
    "Final Answer: <respuesta final al usuario>"
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un agente ReAct especializado en los **reglamentos de Memoria de Título y Práctica Profesional** de la UdeC .\n"
     "Cuando entregues la respuesta final, no debe superar 1800 caracteres. "
     "Responde siempre de manera **detallada y cordial**, considerando lo necesario para la tarea.\n"
     "Piensa paso a paso DENTRO de <think>...</think> como tu scratchpad privado.\n"
     "NUNCA muestres ni cites nada dentro de <think>...</think>.\n\n"
     "Tienes acceso a estas herramientas:\n{tools}\n\n"
     "Si la pregunta lo requiere debes responder usando las herramientas disponibles ({tool_names}).\n"
     "Si la pregunta no se relaciona con estos reglamentos o no puede ser respondida con esas herramientas, "
     "debes contestar EXACTAMENTE con la frase:\n"
     "\"Lo siento, no estoy capacitado para responder preguntas fuera del ámbito de la memoria de título y la práctica profesional.\" \n\n"
     "Si la consulta es small talk (saludos/despedidas), responde de forma breve y cordial sin usar herramientas.\n"
     "No inventes; si falta info, dilo. Finaliza preguntando en qué puedes ayudar.\n\n"
     "{format_instructions}\n"
     "IMPORTANTE: No uses bloques de código ni ```tool_code``` para llamar herramientas."
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
    memory = ConversationBufferMemory(
        memory_key="chat_history",         # Debe coincidir con MessagesPlaceholder
        return_messages=True,
        input_key="input",
        output_key="output",
    )
    executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        memory=memory,                     # Memoria conectada
        verbose=True,
        handle_parsing_errors=True
    )
    return executor

def get_executor_for_user(user_id: int) -> AgentExecutor:
    if user_id not in user_executors:
        user_executors[user_id] = make_executor()   # Cada usuario tiene su propio executor+memoria
    return user_executors[user_id]

# --------------------------------------------------------------------------------------
# Discord events
# --------------------------------------------------------------------------------------
@bot.event
async def on_ready():
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
        output = (res.get("output", "") or "").strip() or "No pude generar una respuesta."

        # Etiqueta SIEMPRE a la persona
        await message.channel.send(f"{message.author.mention} 📘 Respuesta:\n{output}")

    except Exception as e:
        logger.exception("Error procesando el mensaje")
        await message.channel.send(f"{message.author.mention} ❌ Error: {e}")

    # Mantén compatibilidad con prefijo !
    await bot.process_commands(message)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    bot.run(TOKEN_KEY)









