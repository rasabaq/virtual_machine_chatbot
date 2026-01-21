import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { Send, Bot, User, Loader2, Sparkles } from "lucide-react";
import "./ChatInterface.css"; // We'll assume specific styles here or use index.css classes

const ChatInterface = () => {
    const [messages, setMessages] = useState([
        { role: "assistant", content: "¡Hola! Soy tu asistente virtual. ¿En qué puedo ayudarte con tu memoria, práctica o electivos?" }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = { role: "user", content: input };
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setLoading(true);

        try {
            const response = await axios.post("http://localhost:8001/api/chat", {
                message: userMsg.content,
                user_id: "user_123" // Demo flow
            });

            const botMsg = {
                role: "assistant",
                content: response.data.response,
                thoughts: response.data.thoughts
            };
            setMessages((prev) => [...prev, botMsg]);
        } catch (error) {
            console.error("Error sending message:", error);
            setMessages((prev) => [...prev, { role: "assistant", content: "Lo siento, hubo un error al procesar tu solicitud." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="chat-container glass">
            <div className="chat-header">
                <div className="logo-container">
                    <Sparkles className="icon-logo" size={24} />
                    <h1 className="gradient-text">Virtual Assistant</h1>
                </div>
                <div className="status-indicator">
                    <span className="dot"></span> Online
                </div>
            </div>

            <div className="messages-area">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message-wrapper ${msg.role} animate-fade-in`}>
                        <div className="avatar">
                            {msg.role === "assistant" ? <Bot size={20} /> : <User size={20} />}
                        </div>
                        <div className="message-content glass">
                            <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                    </div>
                ))}
                {loading && (
                    <div className="message-wrapper assistant animate-fade-in">
                        <div className="avatar"><Bot size={20} /></div>
                        <div className="message-content glass thinking">
                            <span className="loading-dots">Pensando</span>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <form onSubmit={sendMessage} className="input-area glass">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Escribe tu pregunta aquí..."
                    disabled={loading}
                />
                <button type="submit" disabled={loading || !input.trim()} className="send-btn">
                    {loading ? <Loader2 className="spin" size={20} /> : <Send size={20} />}
                </button>
            </form>
        </div>
    );
};

export default ChatInterface;
