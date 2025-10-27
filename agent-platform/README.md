# AI Agent Platform

A full-stack platform for building, configuring, and deploying conversational AI agents with LLM models. Built with FastAPI, LangChain, React, and TypeScript.

## 🎯 Features

### MVP (Current Version)

- ✅ **User Authentication**: Secure JWT-based authentication with registration and login
- ✅ **Agent Management**: Create, configure, edit, and delete AI agents
- ✅ **Chat Interface**: ChatGPT-like conversational UI with real-time streaming responses
- ✅ **Agent Studio**: Visual configuration interface for agent parameters
- ✅ **Multi-Model Support**: Compatible with OpenAI (GPT-4, GPT-3.5) and Anthropic (Claude) models
- ✅ **Conversation History**: Persistent storage of all conversations and messages
- ✅ **Admin Panel**: LLM provider configuration management (API-based)
- ✅ **Responsive Design**: Works on desktop, tablet, and mobile devices
- ✅ **Markdown Support**: Rich text rendering with code syntax highlighting

## 🏗️ Architecture

### Backend Stack
- **Framework**: FastAPI 0.115.0 (async/await support)
- **Database**: SQLAlchemy 2.0.35 + SQLite (production-ready for PostgreSQL)
- **AI Framework**: LangChain 0.3.7 + LangChain Core 0.3.15
- **Authentication**: JWT (python-jose) + bcrypt (passlib)
- **Streaming**: Server-Sent Events for real-time AI responses
- **API Documentation**: Auto-generated Swagger UI and ReDoc

### Frontend Stack
- **Framework**: React 18.3.1 + TypeScript 5.6.2
- **UI Library**: Ant Design 5.21.4
- **State Management**: Zustand 5.0.0 (lightweight, hook-based)
- **Build Tool**: Vite 5.4.8 (fast HMR and optimized builds)
- **HTTP Client**: Axios 1.7.7
- **Markdown**: react-markdown 9.0.1 + remark-gfm

### LLM Integration
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus
- **Extensible**: Easy to add new providers (Ollama, Azure OpenAI, etc.)

## 📦 Project Structure

```
agent-platform/
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── api/                     # API Layer
│   │   │   └── v1/
│   │   │       ├── agents.py        # Agent CRUD endpoints
│   │   │       ├── auth.py          # Authentication endpoints
│   │   │       ├── chat.py          # Chat & conversation endpoints
│   │   │       └── llm_configs.py   # LLM configuration (admin)
│   │   ├── core/                    # Core Infrastructure
│   │   │   ├── config.py            # App configuration (env vars)
│   │   │   ├── database.py          # Database connection & session
│   │   │   ├── deps.py              # Dependency injection
│   │   │   └── security.py          # JWT & password utilities
│   │   ├── models/                  # SQLAlchemy Models
│   │   │   ├── agent.py             # Agent configuration
│   │   │   ├── conversation.py      # Conversation & Message
│   │   │   ├── llm_config.py        # LLM provider configs
│   │   │   └── user.py              # User accounts
│   │   ├── schemas/                 # Pydantic Schemas
│   │   │   ├── agent.py             # Agent request/response
│   │   │   ├── conversation.py      # Chat request/response
│   │   │   ├── llm_config.py        # Config request/response
│   │   │   └── user.py              # User & token schemas
│   │   ├── services/                # Business Logic
│   │   │   ├── agent/
│   │   │   │   └── agent_executor.py  # LangChain agent execution
│   │   │   └── llm/
│   │   │       └── llm_service.py    # LLM instance creation
│   │   └── main.py                  # FastAPI application
│   ├── requirements.txt             # Python dependencies
│   └── .env.example                 # Environment variables template
│
├── frontend/                         # React Frontend
│   ├── src/
│   │   ├── components/              # Reusable Components
│   │   │   └── MainLayout.tsx       # App layout with navigation
│   │   ├── pages/                   # Page Components
│   │   │   ├── Auth/
│   │   │   │   └── LoginPage.tsx    # Login & registration
│   │   │   ├── Chat/
│   │   │   │   ├── ChatPage.tsx     # Main chat interface
│   │   │   │   └── components/
│   │   │   │       ├── AgentSelector.tsx    # Agent selection UI
│   │   │   │       ├── ChatWindow.tsx       # Message display & input
│   │   │   │       └── ConversationList.tsx # Conversation sidebar
│   │   │   └── Studio/
│   │   │       └── StudioPage.tsx   # Agent management UI
│   │   ├── services/
│   │   │   └── api.ts               # API client (Axios)
│   │   ├── stores/
│   │   │   └── authStore.ts         # Auth state (Zustand)
│   │   ├── types/
│   │   │   └── index.ts             # TypeScript definitions
│   │   ├── App.tsx                  # Root component
│   │   └── main.tsx                 # Entry point
│   ├── package.json                 # Node dependencies
│   ├── tsconfig.json                # TypeScript config
│   ├── vite.config.ts               # Vite config
│   └── index.html                   # HTML template
│
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 18 or higher
- **Package Manager**: npm (comes with Node.js)
- **API Keys**: OpenAI or Anthropic API key (optional for testing)

### Step 1: Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd agent-platform/backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv

   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create environment configuration:**
   ```bash
   # Copy the example file
   cp .env.example .env

   # Generate a secure SECRET_KEY
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

5. **Edit `.env` file** with your SECRET_KEY:
   ```env
   SECRET_KEY=<paste-generated-key-here>
   DATABASE_URL=sqlite:///./agent_platform.db
   CHROMA_PERSIST_DIR=./chroma_data
   CORS_ORIGINS=http://localhost:3000,http://localhost:5173
   ```

6. **Start the backend server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   You should see:
   ```
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   INFO:     Started reloader process
   INFO:     Started server process
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   ```

7. **Verify backend is running:**
   - API Health: http://localhost:8000/health
   - Swagger Docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Step 2: Frontend Setup

1. **Open a new terminal** and navigate to frontend directory:
   ```bash
   cd agent-platform/frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

   This will install all required packages including React, TypeScript, Ant Design, etc.

3. **Start the development server:**
   ```bash
   npm run dev
   ```

   You should see:
   ```
   VITE v5.4.8  ready in XXX ms

   ➜  Local:   http://localhost:3000/
   ➜  Network: use --host to expose
   ➜  press h + enter to show help
   ```

4. **Open the application:**
   - Visit: http://localhost:3000
   - You should see the login page

### Step 3: First-Time Setup

1. **Create an account:**
   - Click "Sign up" link on the login page
   - Enter email, username, and password
   - Click "Sign In" (you'll be redirected to register first)
   - After registration, log in with your credentials

2. **Set yourself as admin (for LLM configuration):**
   ```bash
   # In the backend directory
   sqlite3 agent_platform.db
   ```

   ```sql
   -- Find your user ID
   SELECT id, username, email, is_superuser FROM users;

   -- Set your user as superuser (replace 1 with your user ID)
   UPDATE users SET is_superuser = 1 WHERE id = 1;

   -- Verify
   SELECT id, username, is_superuser FROM users;

   -- Exit
   .quit
   ```

3. **Configure LLM Provider (via API):**

   Get your access token:
   - Log in to the application
   - Open browser DevTools (F12)
   - Go to Application/Storage → Local Storage
   - Copy the value of `access_token`

   Add OpenAI configuration:
   ```bash
   curl -X POST http://localhost:8000/api/v1/llm-configs/ \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN_HERE" \
     -H "Content-Type: application/json" \
     -d '{
       "provider": "openai",
       "display_name": "OpenAI",
       "api_key": "sk-your-openai-api-key-here"
     }'
   ```

   Or add Anthropic configuration:
   ```bash
   curl -X POST http://localhost:8000/api/v1/llm-configs/ \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN_HERE" \
     -H "Content-Type: application/json" \
     -d '{
       "provider": "anthropic",
       "display_name": "Anthropic",
       "api_key": "sk-ant-your-anthropic-key-here"
     }'
   ```

## 📖 User Guide

### Creating Your First Agent

1. **Navigate to Studio:**
   - Click "Studio" in the top navigation bar

2. **Create a new agent:**
   - Click the "Create Agent" button
   - Fill in the form:
     - **Name**: e.g., "Python Coding Assistant"
     - **Description**: e.g., "Helps with Python programming questions"
     - **System Prompt**:
       ```
       You are an expert Python developer. Help users write clean,
       efficient Python code. Provide examples and explain concepts clearly.
       ```
     - **Model Provider**: Select "openai" or "anthropic"
     - **Model Name**:
       - OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
       - Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
     - **Temperature**:
       - 0.0-0.3 for precise/factual responses
       - 0.7 for balanced creativity (default)
       - 1.0-2.0 for more creative responses
     - **Max Tokens**: 2000 (default), up to 32000

3. **Save the agent:**
   - Click "OK" to create the agent
   - The agent will appear in the table

### Starting a Conversation

1. **Navigate to Chat:**
   - Click "Chat" in the top navigation

2. **Start a new conversation:**
   - Click "New Chat" button
   - Select an agent from the grid
   - The chat interface will open

3. **Chat with your agent:**
   - Type your message in the input box at the bottom
   - Press Enter or click "Send"
   - Watch the AI response stream in real-time
   - Continue the conversation with context from previous messages

### Managing Conversations

- **Switch conversations**: Click on any conversation in the left sidebar
- **Delete conversations**: Click the trash icon next to a conversation
- **Conversation titles**: Auto-generated (can be customized in future versions)

### Editing Agents

1. Go to Studio page
2. Click "Edit" on any agent
3. Modify any parameters
4. Click "OK" to save changes
5. Changes take effect immediately in new conversations

## 🔧 Configuration

### Backend Environment Variables

Create `backend/.env` file (see `.env.example`):

```env
# Application Settings
PROJECT_NAME=AI Agent Platform
VERSION=0.1.0
API_V1_STR=/api/v1

# Security (CHANGE THIS IN PRODUCTION!)
SECRET_KEY=your-secret-key-here-use-python-secrets-token-hex-32
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080  # 7 days

# Database
DATABASE_URL=sqlite:///./agent_platform.db
# For PostgreSQL (production):
# DATABASE_URL=postgresql://username:password@localhost:5432/agent_platform

# Vector Database (for future RAG features)
CHROMA_PERSIST_DIR=./chroma_data

# CORS (comma-separated list of allowed origins)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# File Upload (for future knowledge base features)
MAX_UPLOAD_SIZE=10485760  # 10MB
UPLOAD_DIR=./uploads
```

### Frontend Configuration

The frontend automatically proxies API requests to `http://localhost:8000` during development.

To change the backend URL, edit `frontend/vite.config.ts`:

```typescript
export default defineConfig({
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://your-backend-url:8000',
        changeOrigin: true,
      },
    },
  },
})
```

## 📚 API Documentation

### Interactive API Docs

Once the backend is running:
- **Swagger UI** (recommended): http://localhost:8000/docs
  - Interactive API testing
  - Request/response examples
  - Schema documentation
- **ReDoc**: http://localhost:8000/redoc
  - Clean, readable documentation

### API Endpoints Overview

#### Authentication (`/api/v1/auth`)
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/register` | Create new user account | No |
| POST | `/login` | Login and get JWT token | No |

#### Agents (`/api/v1/agents`)
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | List all user's agents | Yes |
| POST | `/` | Create new agent | Yes |
| GET | `/{id}` | Get agent details | Yes |
| PUT | `/{id}` | Update agent | Yes (owner only) |
| DELETE | `/{id}` | Delete agent | Yes (owner only) |

#### Chat (`/api/v1/chat`)
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/conversations` | List user's conversations | Yes |
| POST | `/conversations` | Create new conversation | Yes |
| GET | `/conversations/{id}` | Get conversation with messages | Yes |
| POST | `/conversations/{id}/messages` | Send message (streaming) | Yes |
| DELETE | `/conversations/{id}` | Delete conversation | Yes |

#### LLM Configs (`/api/v1/llm-configs`) - Admin Only
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | List all LLM configs | Yes (admin) |
| POST | `/` | Add LLM provider | Yes (admin) |
| PUT | `/{id}` | Update config | Yes (admin) |
| DELETE | `/{id}` | Delete config | Yes (admin) |

### Example API Calls

#### Register a User
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "testuser",
    "password": "securepassword123"
  }'
```

#### Login
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "securepassword123"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### Create an Agent
```bash
curl -X POST http://localhost:8000/api/v1/agents/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Python Expert",
    "description": "Python programming assistant",
    "system_prompt": "You are a Python expert. Help with code and best practices.",
    "model_provider": "openai",
    "model_name": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 2000
  }'
```

## 🛠️ Development

### Backend Development

```bash
cd backend

# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run with auto-reload (for development)
uvicorn app.main:app --reload

# Run tests (when implemented)
pytest tests/

# Code formatting
black app/

# Linting
ruff check app/

# Type checking
mypy app/
```

### Frontend Development

```bash
cd frontend

# Start dev server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint TypeScript/TSX files
npm run lint

# Type checking
tsc --noEmit
```

### Database Management

#### View Database
```bash
cd backend
sqlite3 agent_platform.db

# List all tables
.tables

# View users
SELECT * FROM users;

# View agents
SELECT id, name, model_provider, model_name FROM agents;

# View conversations
SELECT id, title, created_at FROM conversations LIMIT 10;

# Exit
.quit
```

#### Reset Database
```bash
cd backend
rm agent_platform.db
# Restart the backend - it will recreate the database
```

#### Migrate to PostgreSQL

1. Install PostgreSQL and create a database:
   ```sql
   CREATE DATABASE agent_platform;
   ```

2. Update `.env`:
   ```env
   DATABASE_URL=postgresql://username:password@localhost:5432/agent_platform
   ```

3. Install PostgreSQL driver:
   ```bash
   pip install psycopg2-binary
   ```

4. Restart the backend - tables will be created automatically

## 🔒 Security Best Practices

### For Production Deployment

1. **SECRET_KEY**:
   ```bash
   # Generate a strong secret key
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

2. **Database**:
   - Use PostgreSQL instead of SQLite
   - Enable SSL connections
   - Regular backups

3. **API Keys**:
   - Store in environment variables
   - Use encryption at rest (implement in `llm_config.py`)
   - Rotate keys regularly

4. **HTTPS**:
   - Use reverse proxy (nginx/caddy)
   - Enable SSL/TLS certificates
   - Force HTTPS redirects

5. **CORS**:
   ```env
   CORS_ORIGINS=https://yourdomain.com
   ```

6. **Rate Limiting**:
   - Implement rate limiting middleware
   - Use Redis for distributed rate limiting

7. **Input Validation**:
   - Already implemented via Pydantic
   - Review custom validation rules

8. **Dependencies**:
   ```bash
   # Keep dependencies updated
   pip list --outdated
   npm outdated
   ```

## 🐛 Troubleshooting

### Backend Issues

#### Database locked error
```
Solution: Close all connections to the database
- Stop all running backend instances
- Delete agent_platform.db and restart
```

#### Module not found
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### Port already in use
```bash
# Find and kill process on port 8000
# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Frontend Issues

#### Dependencies install fails
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### Port 3000 in use
```bash
# Edit vite.config.ts and change port
server: {
  port: 3001,  // Change to any available port
}
```

#### API calls fail (CORS errors)
```
Solution: Verify backend CORS_ORIGINS includes your frontend URL
- Check backend/.env
- Restart backend after changes
```

### Common Runtime Issues

#### Agent responses not streaming
```
- Check browser console for errors
- Verify LLM config exists in database
- Check API key is valid
```

#### Login fails after registration
```
- Clear browser localStorage
- Check backend logs for errors
- Verify database connection
```

## 🎨 Customization

### Adding a New LLM Provider

1. **Update `backend/app/services/llm/llm_service.py`:**

```python
from langchain_yourprovider import ChatYourProvider

class LLMService:
    @staticmethod
    def create_llm(agent: Agent, llm_config: LLMConfig | None = None) -> BaseChatModel:
        provider = agent.model_provider.lower()

        # ... existing code ...

        elif provider == "yourprovider":
            return ChatYourProvider(
                model=agent.model_name,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
                api_key=llm_config.api_key if llm_config else None,
            )
```

2. **Update available models:**

```python
@staticmethod
def get_available_models() -> dict[str, list[str]]:
    return {
        "openai": ["gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-3-5-sonnet-20241022"],
        "yourprovider": ["model-1", "model-2"],  # Add this
    }
```

3. **Update frontend model provider options in `StudioPage.tsx`:**

```typescript
<Select.Option value="yourprovider">Your Provider</Select.Option>
```

### Customizing UI Theme

Edit `frontend/src/App.tsx`:

```typescript
<ConfigProvider
  theme={{
    token: {
      colorPrimary: '#1890ff',     // Primary color
      colorSuccess: '#52c41a',     // Success color
      colorWarning: '#faad14',     // Warning color
      colorError: '#ff4d4f',       // Error color
      borderRadius: 8,             // Border radius
      fontSize: 14,                // Base font size
    },
  }}
>
```

### Adding New Agent Parameters

1. Update `backend/app/models/agent.py`
2. Update `backend/app/schemas/agent.py`
3. Update `frontend/src/types/index.ts`
4. Update `frontend/src/pages/Studio/StudioPage.tsx`
5. Restart both backend and frontend

## 🚧 Roadmap

### Phase 2: Enhanced Features (Next)
- [ ] Knowledge Base (RAG)
  - Document upload (PDF, TXT, MD, DOCX)
  - Vector embeddings with ChromaDB
  - Semantic search
  - Document management UI
- [ ] Custom Tools
  - OpenAPI/Swagger import
  - Manual API configuration
  - Function calling
  - Tool execution logs
- [ ] Agent Debugging
  - Execution trace viewer
  - Tool call inspection
  - Step-by-step reasoning display
- [ ] Enhanced Chat UI
  - Message reactions
  - Conversation search
  - Export conversations
  - Share conversations

### Phase 3: Enterprise Features (Future)
- [ ] Admin Dashboard
  - Usage analytics
  - Cost monitoring
  - User management
  - System health metrics
- [ ] Multi-tenancy
  - Organization support
  - Team collaboration
  - Role-based access control
- [ ] Advanced Features
  - Multi-agent workflows
  - Scheduled agents
  - Webhooks
  - API rate limiting
- [ ] Deployment
  - Docker Compose setup
  - Kubernetes manifests
  - CI/CD pipelines
  - Monitoring & logging

## 📊 Data Models

### User
- ID, email, username, hashed_password
- is_active, is_superuser
- created_at, updated_at

### Agent
- ID, name, description, system_prompt
- model_provider, model_name
- temperature, max_tokens
- is_published, owner_id
- created_at, updated_at

### Conversation
- ID, title, user_id, agent_id
- created_at, updated_at
- messages (relationship)

### Message
- ID, conversation_id, role, content
- created_at

### LLMConfig
- ID, provider, display_name
- api_key, api_base
- is_active
- created_at, updated_at

## 🤝 Contributing

Contributions are welcome! This is an MVP implementation with lots of room for improvement.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests (when test framework is added)
5. Submit a pull request

### Code Style
- Backend: Follow PEP 8, use type hints, add docstrings
- Frontend: Follow TypeScript best practices, use functional components

## 📄 License

MIT License - feel free to use this project for personal or commercial purposes.

## 🙋 Support

### Getting Help
1. Check the [API Documentation](http://localhost:8000/docs)
2. Review code comments and docstrings
3. Check the Troubleshooting section above
4. Open an issue in the repository

### Reporting Issues
When reporting issues, include:
- Operating system and version
- Python and Node.js versions
- Error messages and stack traces
- Steps to reproduce

## 🎉 Acknowledgments

Built with amazing open-source tools:

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [LangChain](https://langchain.com/) - LLM application framework
- [React](https://react.dev/) - UI library
- [Ant Design](https://ant.design/) - Enterprise UI components
- [Vite](https://vitejs.dev/) - Next-generation frontend tooling
- [SQLAlchemy](https://www.sqlalchemy.org/) - SQL toolkit and ORM
- [Zustand](https://github.com/pmndrs/zustand) - State management

---

**Built with ❤️ for the AI Agent community**
