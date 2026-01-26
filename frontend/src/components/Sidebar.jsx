import React, { useState, useEffect } from "react";
import { Home, MessageSquare, Clock, Search, ChevronUp, Plus, LogOut, Trash2 } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import api from "../services/api";
import "./Sidebar.css";

const Sidebar = ({ selectedConversation, onSelectConversation, onNewChat, refreshTrigger }) => {
    const { user, logout } = useAuth();
    const [conversations, setConversations] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState("");

    // Fetch conversations from API
    useEffect(() => {
        fetchConversations();
    }, [refreshTrigger]);

    const fetchConversations = async () => {
        try {
            const response = await api.get('/conversations');
            setConversations(response.data);
        } catch (error) {
            console.error("Error fetching conversations:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleDeleteConversation = async (e, conversationId) => {
        e.stopPropagation();
        if (!confirm("¿Estás seguro de que deseas eliminar esta conversación?")) return;

        try {
            await api.delete(`/conversations/${conversationId}`);
            setConversations(prev => prev.filter(c => c.id !== conversationId));
            if (selectedConversation?.id === conversationId) {
                onNewChat();
            }
        } catch (error) {
            console.error("Error deleting conversation:", error);
        }
    };

    // Group conversations by date
    const groupConversations = () => {
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);
        const lastWeek = new Date(today);
        lastWeek.setDate(lastWeek.getDate() - 7);

        const groups = {
            today: [],
            yesterday: [],
            lastWeek: [],
            older: []
        };

        conversations
            .filter(c => c.title.toLowerCase().includes(searchQuery.toLowerCase()))
            .forEach(conv => {
                const convDate = new Date(conv.updated_at);
                if (convDate >= today) {
                    groups.today.push(conv);
                } else if (convDate >= yesterday) {
                    groups.yesterday.push(conv);
                } else if (convDate >= lastWeek) {
                    groups.lastWeek.push(conv);
                } else {
                    groups.older.push(conv);
                }
            });

        return groups;
    };

    const groups = groupConversations();

    return (
        <aside className="sidebar">
            {/* Logo */}
            <div className="sidebar-logo">
                <div className="logo-icon">🎓</div>
                <span className="logo-text">Asistente DII UdeC</span>
            </div>

            {/* New Chat Button */}
            <button className="new-chat-sidebar-btn" onClick={onNewChat}>
                <Plus size={18} />
                <span>Nueva Conversación</span>
            </button>

            {/* Search */}
            <div className="sidebar-search">
                <Search size={16} className="search-icon" />
                <input
                    type="text"
                    placeholder="Buscar conversaciones..."
                    className="search-input"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                />
            </div>

            {/* Conversation History */}
            <div className="prompt-history">
                {loading ? (
                    <div className="history-loading">Cargando...</div>
                ) : conversations.length === 0 ? (
                    <div className="history-empty">
                        <MessageSquare size={24} />
                        <p>Sin conversaciones aún</p>
                        <span>Inicia una nueva conversación</span>
                    </div>
                ) : (
                    <>
                        {groups.today.length > 0 && (
                            <div className="history-section">
                                <h4 className="history-heading">Hoy</h4>
                                {groups.today.map((conv) => (
                                    <button
                                        key={conv.id}
                                        className={`history-item ${selectedConversation?.id === conv.id ? 'active' : ''}`}
                                        onClick={() => onSelectConversation(conv)}
                                    >
                                        <span className="history-item-text">{conv.title}</span>
                                        <button
                                            className="delete-btn"
                                            onClick={(e) => handleDeleteConversation(e, conv.id)}
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </button>
                                ))}
                            </div>
                        )}

                        {groups.yesterday.length > 0 && (
                            <div className="history-section">
                                <h4 className="history-heading">Ayer</h4>
                                {groups.yesterday.map((conv) => (
                                    <button
                                        key={conv.id}
                                        className={`history-item ${selectedConversation?.id === conv.id ? 'active' : ''}`}
                                        onClick={() => onSelectConversation(conv)}
                                    >
                                        <span className="history-item-text">{conv.title}</span>
                                        <button
                                            className="delete-btn"
                                            onClick={(e) => handleDeleteConversation(e, conv.id)}
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </button>
                                ))}
                            </div>
                        )}

                        {groups.lastWeek.length > 0 && (
                            <div className="history-section">
                                <h4 className="history-heading">Últimos 7 días</h4>
                                {groups.lastWeek.map((conv) => (
                                    <button
                                        key={conv.id}
                                        className={`history-item ${selectedConversation?.id === conv.id ? 'active' : ''}`}
                                        onClick={() => onSelectConversation(conv)}
                                    >
                                        <span className="history-item-text">{conv.title}</span>
                                        <button
                                            className="delete-btn"
                                            onClick={(e) => handleDeleteConversation(e, conv.id)}
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </button>
                                ))}
                            </div>
                        )}

                        {groups.older.length > 0 && (
                            <div className="history-section">
                                <h4 className="history-heading">Anteriores</h4>
                                {groups.older.map((conv) => (
                                    <button
                                        key={conv.id}
                                        className={`history-item ${selectedConversation?.id === conv.id ? 'active' : ''}`}
                                        onClick={() => onSelectConversation(conv)}
                                    >
                                        <span className="history-item-text">{conv.title}</span>
                                        <button
                                            className="delete-btn"
                                            onClick={(e) => handleDeleteConversation(e, conv.id)}
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </button>
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>

            {/* User Profile */}
            <div className="user-profile">
                <div className="profile-avatar">
                    <img
                        src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${user?.email || 'user'}`}
                        alt="Avatar"
                    />
                </div>
                <div className="profile-info">
                    <span className="profile-name">{user?.name || 'User'}</span>
                    <span className="profile-email">{user?.email || ''}</span>
                </div>
                <button className="profile-logout" onClick={logout} title="Cerrar sesión">
                    <LogOut size={16} />
                </button>
            </div>
        </aside>
    );
};

export default Sidebar;
