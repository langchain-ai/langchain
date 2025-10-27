/**
 * Conversation List component for displaying and managing conversations.
 */

import React from 'react';
import { List, Button, Popconfirm } from 'antd';
import { DeleteOutlined, MessageOutlined } from '@ant-design/icons';
import { Conversation } from '@/types';
import './ConversationList.css';

interface ConversationListProps {
  conversations: Conversation[];
  selectedId?: number;
  onSelect: (conversation: Conversation) => void;
  onDelete: (id: number) => void;
}

const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  selectedId,
  onSelect,
  onDelete,
}) => {
  return (
    <List
      className="conversation-list"
      dataSource={conversations}
      renderItem={(conversation) => (
        <List.Item
          className={`conversation-item ${
            selectedId === conversation.id ? 'selected' : ''
          }`}
          onClick={() => onSelect(conversation)}
          actions={[
            <Popconfirm
              title="Delete this conversation?"
              onConfirm={(e) => {
                e?.stopPropagation();
                onDelete(conversation.id);
              }}
              okText="Yes"
              cancelText="No"
            >
              <Button
                type="text"
                size="small"
                icon={<DeleteOutlined />}
                onClick={(e) => e.stopPropagation()}
                danger
              />
            </Popconfirm>,
          ]}
        >
          <List.Item.Meta
            avatar={<MessageOutlined />}
            title={conversation.title}
            description={new Date(conversation.updated_at).toLocaleDateString()}
          />
        </List.Item>
      )}
    />
  );
};

export default ConversationList;
