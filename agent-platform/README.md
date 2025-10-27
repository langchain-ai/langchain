# AI Agent Platform

A full-stack platform for building, configuring, and deploying conversational AI agents with LLM models, knowledge bases, and custom tools.

## 🎯 Features

### MVP (Current Version)

- **User Authentication**: JWT-based authentication with registration and login
- **Agent Management**: Create, configure, and manage AI agents with different LLM models
- **Chat Interface**: ChatGPT-like conversational interface with streaming responses
- **Agent Studio**: Configure agents with custom system prompts, model selection, and parameters
- **Multi-Model Support**: Compatible with OpenAI and Anthropic models
- **Conversation History**: Persistent conversation storage and management
- **Admin Panel**: LLM configuration management for administrators

## 🏗️ Architecture

### Backend (FastAPI)
- **Framework**: FastAPI with async support
- **Database**: SQLAlchemy with SQLite (easily upgradeable to PostgreSQL)
- **AI Framework**: LangChain for agent orchestration
- **Authentication**: JWT tokens with bcrypt password hashing
- **Streaming**: Server-Sent Events (SSE) for real-time responses

### Frontend (React + TypeScript)
- **Framework**: React 18 with TypeScript
- **UI Library**: Ant Design
- **State Management**: Zustand
- **Build Tool**: Vite
- **Markdown**: react-markdown for rich message rendering

## 📦 Project Structure

```
agent-platform/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # API endpoints
│   │   ├── core/            # Core configuration
│   │   ├── models/          # Database models
│   │   ├── schemas/         # Pydantic schemas
│   │   └── services/        # Business logic
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/           # Page components
│   │   ├── components/      # Reusable components
│   │   ├── services/        # API client
│   │   ├── stores/          # State management
│   │   └── types/           # TypeScript types
│   └── package.json
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
```bash
cd agent-platform/backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
cat > .env << EOF
SECRET_KEY=your-secret-key-change-this-in-production
DATABASE_URL=sqlite:///./agent_platform.db
CHROMA_PERSIST_DIR=./chroma_data
EOF
```

5. Start the backend server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
API documentation (Swagger): `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd agent-platform/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 📖 Usage Guide

### 1. Register an Account

1. Open `http://localhost:3000/login`
2. Click "Sign up" to create a new account
3. Enter your email, username, and password

### 2. Configure LLM Providers (Admin Only)

First, you need to manually set a user as admin in the database:

```bash
# In the backend directory
sqlite3 agent_platform.db
```

```sql
-- Set user as superuser (replace 1 with your user ID)
UPDATE users SET is_superuser = 1 WHERE id = 1;
.quit
```

Then, in the application:
1. Navigate to the Admin panel (future feature - currently via API)
2. Add LLM configurations with API keys:

```bash
# Using curl to add OpenAI configuration
curl -X POST http://localhost:8000/api/v1/llm-configs/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "display_name": "OpenAI",
    "api_key": "sk-your-openai-api-key"
  }'
```

### 3. Create an Agent

1. Navigate to the "Studio" page
2. Click "Create Agent"
3. Fill in the agent details:
   - **Name**: Give your agent a name (e.g., "Customer Support Bot")
   - **Description**: Brief description of the agent's purpose
   - **System Prompt**: Define the agent's behavior and personality
   - **Model Provider**: Select "openai" or "anthropic"
   - **Model Name**: e.g., "gpt-4o", "claude-3-5-sonnet-20241022"
   - **Temperature**: 0.0 (precise) to 2.0 (creative)
   - **Max Tokens**: Maximum response length (e.g., 2000)
4. Click "OK" to create the agent

### 4. Start a Conversation

1. Navigate to the "Chat" page
2. Click "New Chat"
3. Select an agent from the list
4. Start chatting with your AI agent!

## 🔧 Configuration

### Backend Environment Variables

Create a `.env` file in the `backend` directory:

```env
# Application
PROJECT_NAME=AI Agent Platform
VERSION=0.1.0
API_V1_STR=/api/v1

# Security
SECRET_KEY=your-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080  # 7 days

# Database
DATABASE_URL=sqlite:///./agent_platform.db
# For PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost/agent_platform

# Vector Database
CHROMA_PERSIST_DIR=./chroma_data

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# File Upload
MAX_UPLOAD_SIZE=10485760  # 10MB in bytes
UPLOAD_DIR=./uploads
```

### Frontend Configuration

The frontend proxies API requests to `http://localhost:8000` by default. To change this, edit `vite.config.ts`:

```typescript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://your-backend-url:8000',
      changeOrigin: true,
    },
  },
},
```

## 🛠️ Development

### Backend Development

```bash
# Run with auto-reload
uvicorn app.main:app --reload

# Run tests (when available)
pytest

# Code formatting
black app/
ruff check app/
```

### Frontend Development

```bash
# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint
npm run lint
```

## 📚 API Documentation

Once the backend is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

#### Authentication
- `POST /api/v1/auth/register` - Register a new user
- `POST /api/v1/auth/login` - Login and get access token

#### Agents
- `GET /api/v1/agents/` - List all agents
- `POST /api/v1/agents/` - Create a new agent
- `GET /api/v1/agents/{id}` - Get agent details
- `PUT /api/v1/agents/{id}` - Update an agent
- `DELETE /api/v1/agents/{id}` - Delete an agent

#### Chat
- `GET /api/v1/chat/conversations` - List conversations
- `POST /api/v1/chat/conversations` - Create a conversation
- `POST /api/v1/chat/conversations/{id}/messages` - Send a message (streaming)

#### LLM Configs (Admin)
- `GET /api/v1/llm-configs/` - List LLM configurations
- `POST /api/v1/llm-configs/` - Add LLM configuration

## 🔒 Security Considerations

### For Production Deployment:

1. **Change the SECRET_KEY**: Use a strong, randomly generated secret key
2. **Use PostgreSQL**: Replace SQLite with PostgreSQL for better performance
3. **Encrypt API Keys**: Implement proper encryption for stored API keys (currently stored as plain text)
4. **HTTPS**: Use HTTPS for all communications
5. **CORS**: Restrict CORS origins to your actual frontend domain
6. **Rate Limiting**: Implement rate limiting to prevent abuse
7. **Input Validation**: Already implemented, but review for your use case

## 🎨 Customization

### Adding a New LLM Provider

1. Update `app/services/llm/llm_service.py`:
```python
elif provider == "your_provider":
    return YourProviderChat(**llm_kwargs)
```

2. Update the available models list:
```python
def get_available_models() -> dict[str, list[str]]:
    return {
        "your_provider": ["model-1", "model-2"],
        # ...
    }
```

### Customizing the UI Theme

Edit `frontend/src/App.tsx`:
```typescript
<ConfigProvider
  theme={{
    token: {
      colorPrimary: '#1890ff',  // Change primary color
      borderRadius: 8,           // Change border radius
    },
  }}
>
```

## 🚧 Roadmap

### Phase 2 Features (Planned)
- Knowledge Base (RAG) with document upload
- Custom Tool Integration (API, Database, Code Execution)
- Workflow Editor for multi-step agents
- Agent Debugging and Execution Trace
- Multi-agent Collaboration

### Phase 3 Features (Planned)
- Cost Monitoring and Usage Analytics
- Audit Logs
- User Role Management
- Agent Marketplace
- Deployment Management

## 🤝 Contributing

Contributions are welcome! This is an MVP implementation and there's plenty of room for improvement.

## 📄 License

MIT License - feel free to use this project for your own purposes.

## 🙋 Support

For issues and questions:
1. Check the API documentation at `http://localhost:8000/docs`
2. Review the code comments in the source files
3. Open an issue in the repository

## 🎉 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://langchain.com/)
- [React](https://react.dev/)
- [Ant Design](https://ant.design/)
- [Vite](https://vitejs.dev/)
