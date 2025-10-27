/**
 * Main App component with routing.
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { useAuthStore } from '@/stores/authStore';
import LoginPage from '@/pages/Auth/LoginPage';
import ChatPage from '@/pages/Chat/ChatPage';
import StudioPage from '@/pages/Studio/StudioPage';
import MainLayout from '@/components/MainLayout';

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuthStore();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

const App: React.FC = () => {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
        },
      }}
    >
      <Router>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <MainLayout />
              </ProtectedRoute>
            }
          >
            <Route index element={<ChatPage />} />
            <Route path="studio" element={<StudioPage />} />
          </Route>
        </Routes>
      </Router>
    </ConfigProvider>
  );
};

export default App;
