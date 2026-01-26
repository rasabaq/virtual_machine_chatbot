import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { Send, ChevronDown, Loader2 } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import api from "../services/api";
import "./Dashboard.css";

const Dashboard = ({ selectedConversation, onNewChat, onConversationCreated }) => {
    const { user } = useAuth();
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [loadingMessages, setLoadingMessages] = useState(false);
    const [currentConversationId, setCurrentConversationId] = useState(null);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Load messages when conversation changes
    useEffect(() => {
        if (selectedConversation) {
            loadConversationMessages(selectedConversation.id);
            setCurrentConversationId(selectedConversation.id);
        } else {
            // New chat - clear messages
            setMessages([]);
            setCurrentConversationId(null);
        }
    }, [selectedConversation]);

    const loadConversationMessages = async (conversationId) => {
        setLoadingMessages(true);
        try {
            const response = await api.get(`/conversations/${conversationId}`);
            setMessages(response.data.messages.map(m => ({
                role: m.role,
                content: m.content
            })));
        } catch (error) {
            console.error("Error loading messages:", error);
        } finally {
            setLoadingMessages(false);
        }
    };

    // Get greeting based on time
    const getGreeting = () => {
        const hour = new Date().getHours();
        if (hour < 12) return "Buenos Días";
        if (hour < 18) return "Buenas Tardes";
        return "Buenas Noches";
    };

    const sendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = { role: "user", content: input };
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setLoading(true);

        try {
            const response = await api.post("/chat", {
                message: userMsg.content,
                conversation_id: currentConversationId
            });

            const botMsg = {
                role: "assistant",
                content: response.data.response
            };
            setMessages((prev) => [...prev, botMsg]);

            // If this was a new chat, notify parent of the new conversation
            if (!currentConversationId) {
                setCurrentConversationId(response.data.conversation_id);
                onConversationCreated({
                    id: response.data.conversation_id,
                    title: userMsg.content.slice(0, 50) + (userMsg.content.length > 50 ? "..." : "")
                });
            }
        } catch (error) {
            console.error("Error sending message:", error);
            setMessages((prev) => [...prev, {
                role: "assistant",
                content: "Lo siento, hubo un error al procesar tu solicitud."
            }]);
        } finally {
            setLoading(false);
        }
    };


    const showChat = messages.length > 0 || loadingMessages;

    return (
        <main className="dashboard">
            {/* Top Bar */}
            <div className="dashboard-topbar">
                <div className="model-selector">
                    <span className="model-icon">🎓</span>
                    <span className="model-name">Asistente DII 5-nano</span>
                    <ChevronDown size={16} />
                </div>

                <div className="topbar-actions">
                    <button className="new-chat-btn" onClick={onNewChat}>
                        <span>+</span> Nueva Conversación
                    </button>
                    <button className="avatar-btn">
                        <img
                            src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${user?.email || 'user'}&backgroundColor=003366`}
                            alt="User"
                        />
                    </button>
                </div>
            </div>

            {/* Main Content */}
            <div className="dashboard-content">
                {!showChat ? (
                    /* Welcome View */
                    <div className="welcome-view animate-fade-in-up">
                        {/* Decorative Blob */}
                        <div className="decorative-blob"></div>

                        {/* Greeting */}
                        <h1 className="greeting-headline">
                            {getGreeting()}, {user?.name || 'User'}
                        </h1>
                        <h2 className="greeting-subtitle">
                            ¿En qué puedo <span className="accent-text">ayudarte hoy?</span>
                        </h2>

                        {/* Chat Input Panel */}
                        <div className="input-panel glass">
                            <div className="input-row">
                                <span className="input-icon">✨</span>
                                <input
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && sendMessage(e)}
                                    placeholder="Escribe tu consulta o mensaje..."
                                    disabled={loading}
                                    className="main-input"
                                />
                            </div>

                            <div className="input-actions">
                                <button
                                    className="send-btn"
                                    onClick={sendMessage}
                                    disabled={loading || !input.trim()}
                                >
                                    {loading ? <Loader2 className="spin" size={20} /> : <Send size={20} />}
                                </button>
                            </div>
                        </div>
                    </div>
                ) : (
                    /* Chat View */
                    <div className="chat-view">
                        <div className="messages-area">
                            {loadingMessages ? (
                                <div className="messages-loading">
                                    <Loader2 className="spin" size={24} />
                                    <span>Cargando mensajes...</span>
                                </div>
                            ) : (
                                <>
                                    {messages.map((msg, idx) => (
                                        <div key={idx} className={`message-wrapper ${msg.role} animate-fade-in`}>
                                            <div className="message-avatar">
                                                {msg.role === "assistant" ? "🎓" : "👤"}
                                            </div>
                                            <div className="message-bubble">
                                                <ReactMarkdown>{msg.content}</ReactMarkdown>
                                            </div>
                                        </div>
                                    ))}
                                    {loading && (
                                        <div className="message-wrapper assistant animate-fade-in">
                                            <div className="message-avatar">🎓</div>
                                            <div className="message-bubble thinking">
                                                <span className="loading-dots">Pensando</span>
                                            </div>
                                        </div>
                                    )}
                                </>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Chat Input */}
                        <form onSubmit={sendMessage} className="chat-input-area glass">
                            <span className="input-icon">✨</span>
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Escribe tu mensaje..."
                                disabled={loading || loadingMessages}
                                className="chat-input"
                            />
                            <button
                                type="submit"
                                disabled={loading || loadingMessages || !input.trim()}
                                className="send-btn small"
                            >
                                {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
                            </button>
                        </form>
                    </div>
                )}
            </div>
        </main>
    );
};

export default Dashboard;
