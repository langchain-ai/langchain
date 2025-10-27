/**
 * Main Layout component with navigation.
 */

import React from 'react';
import { Layout, Menu, Button, Dropdown } from 'antd';
import {
  MessageOutlined,
  RobotOutlined,
  UserOutlined,
  LogoutOutlined,
} from '@ant-design/icons';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import type { MenuProps } from 'antd';
import './MainLayout.css';

const { Header, Content } = Layout;

const MainLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { logout } = useAuthStore();

  const menuItems: MenuProps['items'] = [
    {
      key: '/',
      icon: <MessageOutlined />,
      label: 'Chat',
      onClick: () => navigate('/'),
    },
    {
      key: '/studio',
      icon: <RobotOutlined />,
      label: 'Studio',
      onClick: () => navigate('/studio'),
    },
  ];

  const userMenuItems: MenuProps['items'] = [
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: 'Logout',
      onClick: () => {
        logout();
        navigate('/login');
      },
    },
  ];

  const selectedKey = location.pathname === '/' ? '/' : location.pathname;

  return (
    <Layout className="main-layout">
      <Header className="main-header">
        <div className="logo">AI Agent Platform</div>
        <Menu
          theme="dark"
          mode="horizontal"
          selectedKeys={[selectedKey]}
          items={menuItems}
          style={{ flex: 1, minWidth: 0 }}
        />
        <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
          <Button type="text" icon={<UserOutlined />} style={{ color: 'white' }}>
            Account
          </Button>
        </Dropdown>
      </Header>
      <Content className="main-content">
        <Outlet />
      </Content>
    </Layout>
  );
};

export default MainLayout;
