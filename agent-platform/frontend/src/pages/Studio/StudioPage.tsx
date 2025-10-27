/**
 * Agent Studio Page component for creating and managing agents.
 */

import React, { useEffect, useState } from 'react';
import {
  Layout,
  Table,
  Button,
  Space,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  message,
  Popconfirm,
  Tag,
} from 'antd';
import { PlusOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons';
import { api } from '@/services/api';
import { Agent, AgentCreateRequest, AgentUpdateRequest } from '@/types';
import './StudioPage.css';

const { Content } = Layout;
const { TextArea } = Input;

const StudioPage: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null);
  const [form] = Form.useForm();

  useEffect(() => {
    loadAgents();
  }, []);

  const loadAgents = async () => {
    setLoading(true);
    try {
      const data = await api.getAgents();
      setAgents(data);
    } catch (error) {
      message.error('Failed to load agents');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = () => {
    setEditingAgent(null);
    form.resetFields();
    setModalVisible(true);
  };

  const handleEdit = (agent: Agent) => {
    setEditingAgent(agent);
    form.setFieldsValue(agent);
    setModalVisible(true);
  };

  const handleDelete = async (id: number) => {
    try {
      await api.deleteAgent(id);
      message.success('Agent deleted');
      loadAgents();
    } catch (error) {
      message.error('Failed to delete agent');
    }
  };

  const handleSubmit = async (values: any) => {
    try {
      if (editingAgent) {
        await api.updateAgent(editingAgent.id, values as AgentUpdateRequest);
        message.success('Agent updated');
      } else {
        await api.createAgent(values as AgentCreateRequest);
        message.success('Agent created');
      }
      setModalVisible(false);
      loadAgents();
    } catch (error) {
      message.error('Failed to save agent');
    }
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: 'Model',
      key: 'model',
      render: (record: Agent) => (
        <Space>
          <Tag color="blue">{record.model_provider}</Tag>
          <Tag>{record.model_name}</Tag>
        </Space>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'is_published',
      key: 'is_published',
      render: (published: boolean) => (
        <Tag color={published ? 'green' : 'default'}>
          {published ? 'Published' : 'Draft'}
        </Tag>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: Agent) => (
        <Space>
          <Button
            type="link"
            icon={<EditOutlined />}
            onClick={() => handleEdit(record)}
          >
            Edit
          </Button>
          <Popconfirm
            title="Delete this agent?"
            onConfirm={() => handleDelete(record.id)}
            okText="Yes"
            cancelText="No"
          >
            <Button type="link" danger icon={<DeleteOutlined />}>
              Delete
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Layout className="studio-page">
      <Content className="studio-content">
        <div className="studio-header">
          <h1>Agent Studio</h1>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={handleCreate}
          >
            Create Agent
          </Button>
        </div>
        <Table
          columns={columns}
          dataSource={agents}
          rowKey="id"
          loading={loading}
        />
        <Modal
          title={editingAgent ? 'Edit Agent' : 'Create Agent'}
          open={modalVisible}
          onCancel={() => setModalVisible(false)}
          onOk={() => form.submit()}
          width={700}
        >
          <Form
            form={form}
            layout="vertical"
            onFinish={handleSubmit}
            initialValues={{
              temperature: 0.7,
              max_tokens: 2000,
              model_provider: 'openai',
            }}
          >
            <Form.Item
              name="name"
              label="Name"
              rules={[{ required: true, message: 'Please enter agent name' }]}
            >
              <Input placeholder="My Assistant" />
            </Form.Item>
            <Form.Item name="description" label="Description">
              <TextArea rows={2} placeholder="Brief description of the agent" />
            </Form.Item>
            <Form.Item
              name="system_prompt"
              label="System Prompt"
              rules={[
                { required: true, message: 'Please enter system prompt' },
              ]}
            >
              <TextArea
                rows={6}
                placeholder="You are a helpful assistant..."
              />
            </Form.Item>
            <Form.Item
              name="model_provider"
              label="Model Provider"
              rules={[{ required: true }]}
            >
              <Select>
                <Select.Option value="openai">OpenAI</Select.Option>
                <Select.Option value="anthropic">Anthropic</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item
              name="model_name"
              label="Model Name"
              rules={[{ required: true }]}
            >
              <Input placeholder="gpt-4o" />
            </Form.Item>
            <Form.Item name="temperature" label="Temperature">
              <InputNumber min={0} max={2} step={0.1} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item name="max_tokens" label="Max Tokens">
              <InputNumber min={1} max={32000} style={{ width: '100%' }} />
            </Form.Item>
            {editingAgent && (
              <Form.Item name="is_published" label="Published" valuePropName="checked">
                <Select>
                  <Select.Option value={true}>Published</Select.Option>
                  <Select.Option value={false}>Draft</Select.Option>
                </Select>
              </Form.Item>
            )}
          </Form>
        </Modal>
      </Content>
    </Layout>
  );
};

export default StudioPage;
