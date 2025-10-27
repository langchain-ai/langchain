/**
 * Login Page component.
 */

import React, { useState } from 'react';
import { Form, Input, Button, Card, Typography, message } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import './AuthPage.css';

const { Title, Paragraph } = Typography;

const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const { login, loading } = useAuthStore();
  const [form] = Form.useForm();

  const handleSubmit = async (values: any) => {
    try {
      await login(values);
      message.success('Login successful');
      navigate('/');
    } catch (error) {
      // Error handled by store
    }
  };

  return (
    <div className="auth-page">
      <Card className="auth-card">
        <div className="auth-header">
          <Title level={2}>AI Agent Platform</Title>
          <Paragraph>Sign in to your account</Paragraph>
        </div>
        <Form form={form} onFinish={handleSubmit} layout="vertical">
          <Form.Item
            name="username"
            rules={[{ required: true, message: 'Please enter your username' }]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder="Username"
              size="large"
            />
          </Form.Item>
          <Form.Item
            name="password"
            rules={[{ required: true, message: 'Please enter your password' }]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Password"
              size="large"
            />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              size="large"
              block
              loading={loading}
            >
              Sign In
            </Button>
          </Form.Item>
        </Form>
        <div className="auth-footer">
          Don't have an account? <Link to="/register">Sign up</Link>
        </div>
      </Card>
    </div>
  );
};

export default LoginPage;
