/**
 * Type definitions for the AI Agent Platform frontend.
 */

export interface User {
  id: number;
  email: string;
  username: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
}

export interface Agent {
  id: number;
  name: string;
  description?: string;
  system_prompt: string;
  model_provider: 'openai' | 'anthropic' | 'custom';
  model_name: string;
  temperature: number;
  max_tokens: number;
  is_published: boolean;
  owner_id: number;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: number;
  conversation_id: number;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
}

export interface Conversation {
  id: number;
  title: string;
  user_id: number;
  agent_id: number;
  created_at: string;
  updated_at: string;
  messages: Message[];
}

export interface LLMConfig {
  id: number;
  provider: string;
  display_name: string;
  api_base?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface AgentCreateRequest {
  name: string;
  description?: string;
  system_prompt: string;
  model_provider: string;
  model_name: string;
  temperature?: number;
  max_tokens?: number;
}

export interface AgentUpdateRequest {
  name?: string;
  description?: string;
  system_prompt?: string;
  model_provider?: string;
  model_name?: string;
  temperature?: number;
  max_tokens?: number;
  is_published?: boolean;
}

export interface ConversationCreateRequest {
  agent_id: number;
  title?: string;
}

export interface ChatRequest {
  message: string;
}
