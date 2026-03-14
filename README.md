# AI Mastery Hub

A world-class graduate-level AI/ML learning platform. Rigorous, research-grade content for engineers and researchers who are past the basics.

## What it is

8-semester curriculum covering the full breadth of modern AI — mathematical foundations, deep learning, reinforcement learning, generative models, LLMs, and frontier topics. Content is written at graduate/researcher level with properly typeset math, annotated code, and inline knowledge checks.

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS
- **Backend**: Express.js + TypeScript
- **Content**: Markdown files on disk (no CMS, no database for content)
- **Progress**: SQLite via better-sqlite3 (anonymous sessions)
- **Workspace**: pnpm monorepo (`frontend/` + `backend/`)

## Dev Setup

```bash
# Install dependencies
pnpm install

# Start both frontend (port 5173) and backend (port 3001)
pnpm dev
```

Frontend proxies `/api/*` to the backend automatically in dev.

## Project Structure

```
AI-Mastery-Hub/
├── frontend/        # React app
├── backend/         # Express API
├── content/         # Markdown lesson files + curriculum.json
├── Research/        # Curriculum research and references
└── docs/            # Spec and planning docs
```

## Spec

Full platform design spec: [`docs/superpowers/specs/2026-03-14-ai-mastery-hub-redesign-design.md`](docs/superpowers/specs/2026-03-14-ai-mastery-hub-redesign-design.md)
