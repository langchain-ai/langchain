/**
 * Main Chat Page component.
 *
 * This component provides a ChatGPT-like interface for users to interact
 * with AI agents.
 */

import React, { useEffect, useState } from 'react';
import { Layout, List, Button, Input, message, Spin, Empty } from 'antd';
import { PlusOutlined, MessageOutlined } from '@ant-design/icons';
import { api } from '@/services/api';
import { Agent, Conversation } from '@/types';
import ConversationList from './components/ConversationList';
import ChatWindow from './components/ChatWindow';
import AgentSelector from './components/AgentSelector';
import './ChatPage.css';

const { Sider, Content } = Layout;

const ChatPage: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [loading, setLoading] = useState(false);
  const [showAgentSelector, setShowAgentSelector] = useState(false);

  useEffect(() => {
    loadAgents();
    loadConversations();
  }, []);

  const loadAgents = async () => {
    try {
      const data = await api.getAgents();
      setAgents(data);
    } catch (error) {
      message.error('Failed to load agents');
    }
  };

  const loadConversations = async () => {
    try {
      const data = await api.getConversations();
      setConversations(data);
    } catch (error) {
      message.error('Failed to load conversations');
    }
  };

  const handleCreateConversation = async (agentId: number) => {
    setLoading(true);
    try {
      const newConversation = await api.createConversation({ agent_id: agentId });
      setConversations([newConversation, ...conversations]);
      setSelectedConversation(newConversation);
      setShowAgentSelector(false);
      message.success('Conversation created');
    } catch (error) {
      message.error('Failed to create conversation');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectConversation = async (conversation: Conversation) => {
    try {
      const fullConversation = await api.getConversation(conversation.id);
      setSelectedConversation(fullConversation);
    } catch (error) {
      message.error('Failed to load conversation');
    }
  };

  const handleDeleteConversation = async (id: number) => {
    try {
      await api.deleteConversation(id);
      setConversations(conversations.filter((c) => c.id !== id));
      if (selectedConversation?.id === id) {
        setSelectedConversation(null);
      }
      message.success('Conversation deleted');
    } catch (error) {
      message.error('Failed to delete conversation');
    }
  };

  const handleNewChat = () => {
    setShowAgentSelector(true);
    setSelectedConversation(null);
  };

  return (
    <Layout className="chat-page">
      <Sider width={280} className="chat-sidebar">
        <div className="sidebar-header">
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={handleNewChat}
            block
          >
            New Chat
          </Button>
        </div>
        <ConversationList
          conversations={conversations}
          selectedId={selectedConversation?.id}
          onSelect={handleSelectConversation}
          onDelete={handleDeleteConversation}
        />
      </Sider>
      <Content className="chat-content">
        {showAgentSelector ? (
          <AgentSelector
            agents={agents}
            onSelect={handleCreateConversation}
            loading={loading}
          />
        ) : selectedConversation ? (
          <ChatWindow
            conversation={selectedConversation}
            onUpdate={loadConversations}
          />
        ) : (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="Select a conversation or start a new chat"
          />
        )}
      </Content>
    </Layout>
  );
};

export default ChatPage;
