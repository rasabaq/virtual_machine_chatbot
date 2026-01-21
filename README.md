# Virtual Machine Chatbot - Web Application

A modern web interface for the Virtual Machine Chatbot, powered by a hybrid architecture.

## Tech Stack

### Backend
- **Django**: Manages Data Models (Interactions) and Admin Interface.
- **FastAPI**: High-performance endpoint for the Agent Chat System.
- **LangChain + Pydantic**: Structured Agent logic with RAG capabilities.
- **SQLite**: Local database.

### Frontend
- **React (Vite)**: Fast, modern UI.
- **Vanilla CSS**: Premium dark-mode design with animations.
- **Lucide React**: Icons.

## Features
- **RAG System**: Answers questions based on `memoriatitulo.txt`, `practicaprofesional.txt`, `electivos.txt`.
- **Reasoning Display**: (Internal) Agent utilizes `<think>` methodology for chain-of-thought.
- **History Tracking**: All interactions are saved to the SQLite database and viewable in Django Admin.

## Quick Start (Windows)
Simply run `start_services.bat` to launch all services.

See [docs/setup.md](docs/setup.md) for manual setup.
