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

llm_thesis = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key= GOOGLE_API_KEY, temperature=0.6)
llm_internship = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key= GOOGLE_API_KEY, temperature=0.6)

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

agent = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)

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
     "Eres un agente ReAct para Ingeniería Civil Industrial UdeC.\n"
     "Cuando entregues la respuesta final, no debe superar 1800 caracteres. "
     "Resume si es necesario.\n"
     "Tienes acceso a estas herramientas:\n{tools}\n\n"
     "Sólo puedes usar estas herramientas por nombre: {tool_names}.\n"
     "Si la consulta es small talk (saludos/despedidas), responde breve sin usar herramientas.\n"
     "No inventes; si falta info, dilo. Finaliza preguntando en qué puedes ayudar de manera coordial.\n\n"
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

react_agent = create_react_agent(agent, tools, prompt)
agent_executor = AgentExecutor(agent=react_agent, tools=tools,memory=memory, verbose=True, handle_parsing_errors=True)

# --- Discord events ---
@bot.event
async def on_ready():
    logger.info(f"✅ Bot conectado como {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Si quieres comandos con prefijo, permite que sigan funcionando
    # (no retornes antes de process_commands si planeas tener !comandos)
    try:
        user_text = message.content.strip()
        if not user_text:
            return

        # Si no hay herramientas (p. ej., faltan archivos), responde directo:
        if not tools:
            await message.channel.send("No tengo fuentes cargadas aún. Sube o configura los documentos necesarios. ¿En qué puedo ayudarte?")
            await bot.process_commands(message)
            return

        # Llama al agente (asíncrono)
        res = await agent_executor.ainvoke({"input": user_text})
        output = res.get("output", "").strip() or "No pude generar una respuesta."
        prefix = "📘 Respuesta:\n"
        if len(output) <= 2000 - len(prefix):
            await message.channel.send(prefix + output)
        else:
            await message.channel.send(prefix + output[:2000 - len(prefix)])
            for i in range(2000 - len(prefix), len(output), 2000):
                await message.channel.send(output[i:i+2000])
    except Exception as e:
        logger.exception("Error procesando el mensaje")
        await message.channel.send(f"❌ Error: {e}")

    # Mantén compatibilidad con prefijo !
    await bot.process_commands(message)

if __name__ == "__main__":
    bot.run(TOKEN_KEY)







