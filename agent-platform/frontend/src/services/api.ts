/**
 * API client for the AI Agent Platform backend.
 */

import axios, { AxiosInstance } from 'axios';
import {
  Agent,
  AgentCreateRequest,
  AgentUpdateRequest,
  ChatRequest,
  Conversation,
  ConversationCreateRequest,
  LoginRequest,
  LLMConfig,
  RegisterRequest,
  TokenResponse,
  User,
} from '@/types';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: '/api/v1',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor to include auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Clear token and redirect to login
          localStorage.removeItem('access_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Authentication
  async login(data: LoginRequest): Promise<TokenResponse> {
    const response = await this.client.post<TokenResponse>('/auth/login', data);
    return response.data;
  }

  async register(data: RegisterRequest): Promise<User> {
    const response = await this.client.post<User>('/auth/register', data);
    return response.data;
  }

  // Agents
  async getAgents(): Promise<Agent[]> {
    const response = await this.client.get<Agent[]>('/agents/');
    return response.data;
  }

  async getAgent(id: number): Promise<Agent> {
    const response = await this.client.get<Agent>(`/agents/${id}`);
    return response.data;
  }

  async createAgent(data: AgentCreateRequest): Promise<Agent> {
    const response = await this.client.post<Agent>('/agents/', data);
    return response.data;
  }

  async updateAgent(id: number, data: AgentUpdateRequest): Promise<Agent> {
    const response = await this.client.put<Agent>(`/agents/${id}`, data);
    return response.data;
  }

  async deleteAgent(id: number): Promise<void> {
    await this.client.delete(`/agents/${id}`);
  }

  // Conversations
  async getConversations(): Promise<Conversation[]> {
    const response = await this.client.get<Conversation[]>('/chat/conversations');
    return response.data;
  }

  async getConversation(id: number): Promise<Conversation> {
    const response = await this.client.get<Conversation>(`/chat/conversations/${id}`);
    return response.data;
  }

  async createConversation(data: ConversationCreateRequest): Promise<Conversation> {
    const response = await this.client.post<Conversation>('/chat/conversations', data);
    return response.data;
  }

  async deleteConversation(id: number): Promise<void> {
    await this.client.delete(`/chat/conversations/${id}`);
  }

  // Chat - Stream response
  async sendMessage(conversationId: number, data: ChatRequest): Promise<ReadableStream> {
    const token = localStorage.getItem('access_token');
    const response = await fetch(`/api/v1/chat/conversations/${conversationId}/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error('Failed to send message');
    }

    if (!response.body) {
      throw new Error('No response body');
    }

    return response.body;
  }

  // LLM Configs (Admin only)
  async getLLMConfigs(): Promise<LLMConfig[]> {
    const response = await this.client.get<LLMConfig[]>('/llm-configs/');
    return response.data;
  }

  async createLLMConfig(data: any): Promise<LLMConfig> {
    const response = await this.client.post<LLMConfig>('/llm-configs/', data);
    return response.data;
  }

  async updateLLMConfig(id: number, data: any): Promise<LLMConfig> {
    const response = await this.client.put<LLMConfig>(`/llm-configs/${id}`, data);
    return response.data;
  }

  async deleteLLMConfig(id: number): Promise<void> {
    await this.client.delete(`/llm-configs/${id}`);
  }
}

export const api = new ApiClient();
