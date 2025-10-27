/**
 * Chat Window component for displaying messages and handling user input.
 */

import React, { useEffect, useRef, useState } from 'react';
import { Input, Button, message as antdMessage, Space } from 'antd';
import { SendOutlined } from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api } from '@/services/api';
import { Conversation, Message } from '@/types';
import './ChatWindow.css';

const { TextArea } = Input;

interface ChatWindowProps {
  conversation: Conversation;
  onUpdate: () => void;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ conversation, onUpdate }) => {
  const [messages, setMessages] = useState<Message[]>(conversation.messages || []);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMessages(conversation.messages || []);
  }, [conversation]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);
    setStreamingMessage('');

    // Add user message to UI
    const newUserMessage: Message = {
      id: Date.now(),
      conversation_id: conversation.id,
      role: 'user',
      content: userMessage,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, newUserMessage]);

    try {
      const stream = await api.sendMessage(conversation.id, { message: userMessage });
      const reader = stream.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        fullResponse += chunk;
        setStreamingMessage(fullResponse);
      }

      // Add assistant message to UI
      const assistantMessage: Message = {
        id: Date.now() + 1,
        conversation_id: conversation.id,
        role: 'assistant',
        content: fullResponse,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setStreamingMessage('');
      onUpdate();
    } catch (error) {
      console.error('Failed to send message:', error);
      antdMessage.error('Failed to send message');
      // Remove the user message if sending failed
      setMessages((prev) => prev.filter((m) => m.id !== newUserMessage.id));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-window">
      <div className="messages-container">
        {messages.map((msg) => (
          <div key={msg.id} className={`message message-${msg.role}`}>
            <div className="message-role">
              {msg.role === 'user' ? 'You' : 'Assistant'}
            </div>
            <div className="message-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            </div>
          </div>
        ))}
        {streamingMessage && (
          <div className="message message-assistant">
            <div className="message-role">Assistant</div>
            <div className="message-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {streamingMessage}
              </ReactMarkdown>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-container">
        <Space.Compact style={{ width: '100%' }}>
          <TextArea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
            autoSize={{ minRows: 1, maxRows: 4 }}
            disabled={loading}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            loading={loading}
            disabled={!input.trim()}
          >
            Send
          </Button>
        </Space.Compact>
      </div>
    </div>
  );
};

export default ChatWindow;
