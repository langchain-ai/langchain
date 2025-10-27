/**
 * Authentication state management using Zustand.
 */

import { create } from 'zustand';
import { api } from '@/services/api';
import { LoginRequest, RegisterRequest, User } from '@/types';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;

  login: (credentials: LoginRequest) => Promise<void>;
  register: (data: RegisterRequest) => Promise<void>;
  logout: () => void;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: localStorage.getItem('access_token'),
  isAuthenticated: !!localStorage.getItem('access_token'),
  loading: false,
  error: null,

  login: async (credentials: LoginRequest) => {
    set({ loading: true, error: null });
    try {
      const response = await api.login(credentials);
      localStorage.setItem('access_token', response.access_token);
      set({
        token: response.access_token,
        isAuthenticated: true,
        loading: false,
      });
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Login failed',
        loading: false,
      });
      throw error;
    }
  },

  register: async (data: RegisterRequest) => {
    set({ loading: true, error: null });
    try {
      const user = await api.register(data);
      set({ user, loading: false });
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Registration failed',
        loading: false,
      });
      throw error;
    }
  },

  logout: () => {
    localStorage.removeItem('access_token');
    set({
      user: null,
      token: null,
      isAuthenticated: false,
    });
  },

  clearError: () => set({ error: null }),
}));
