/**
 * Agent Selector component for choosing an agent to start a conversation.
 */

import React from 'react';
import { Card, Row, Col, Typography, Tag, Empty } from 'antd';
import { RobotOutlined } from '@ant-design/icons';
import { Agent } from '@/types';
import './AgentSelector.css';

const { Title, Paragraph } = Typography;

interface AgentSelectorProps {
  agents: Agent[];
  onSelect: (agentId: number) => void;
  loading: boolean;
}

const AgentSelector: React.FC<AgentSelectorProps> = ({
  agents,
  onSelect,
  loading,
}) => {
  if (agents.length === 0) {
    return (
      <div className="agent-selector-empty">
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="No agents available. Create an agent first in the Studio."
        />
      </div>
    );
  }

  return (
    <div className="agent-selector">
      <Title level={3}>Select an Agent</Title>
      <Paragraph>Choose an agent to start a conversation</Paragraph>
      <Row gutter={[16, 16]}>
        {agents.map((agent) => (
          <Col xs={24} sm={12} md={8} lg={6} key={agent.id}>
            <Card
              hoverable
              onClick={() => !loading && onSelect(agent.id)}
              className="agent-card"
            >
              <div className="agent-icon">
                <RobotOutlined style={{ fontSize: 48 }} />
              </div>
              <Title level={5}>{agent.name}</Title>
              <Paragraph ellipsis={{ rows: 2 }}>
                {agent.description || 'No description'}
              </Paragraph>
              <div className="agent-meta">
                <Tag color="blue">{agent.model_provider}</Tag>
                <Tag>{agent.model_name}</Tag>
              </div>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};

export default AgentSelector;
