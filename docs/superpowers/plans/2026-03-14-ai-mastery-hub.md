# AI Mastery Hub Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a graduate-level AI/ML learning platform with markdown-based lessons, KaTeX math, inline quizzes, and SQLite progress tracking — content quality first.

**Architecture:** React 18 + Vite frontend (port 5173) proxied to Express + TypeScript backend (port 3001). Content lives as markdown files in `content/` at repo root. Progress stored in SQLite via better-sqlite3. Anonymous sessions via localStorage UUID. No database for content.

**Tech Stack:** React 18, TypeScript, Vite 5, Tailwind CSS v3, React Router v6, TanStack Query v5, react-markdown + remark-math + rehype-katex, react-syntax-highlighter, Express.js, better-sqlite3, js-yaml, Vitest

---

## Pre-flight: Repo State

> **Note for executor:** The root `package.json` and `pnpm-workspace.yaml` already exist in the repo (pre-created during brainstorming). Verify before proceeding:
>
> ```bash
> cat package.json          # should have "dev": "concurrently ..." script
> cat pnpm-workspace.yaml   # should list frontend and backend
> ```
>
> If they exist and are correct, skip any root config creation. The repo currently contains only: `docs/`, `Research/`, `package.json`, `pnpm-workspace.yaml`, `README.md` — no `frontend/`, `backend/`, or `content/` yet. This is a greenfield build.

---

## Chunk 1: Backend Foundation

### Task 1: Scaffold backend package

**Files:**
- Create: `backend/package.json`
- Create: `backend/tsconfig.json`
- Create: `backend/.env.example`
- Create: `backend/src/app.ts`

- [ ] **Step 1: Create `backend/package.json`**

```json
{
  "name": "backend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "tsx watch src/app.ts",
    "build": "tsc",
    "start": "node dist/app.js",
    "test": "vitest run"
  },
  "dependencies": {
    "better-sqlite3": "^9.4.3",
    "cors": "^2.8.5",
    "express": "^4.19.2",
    "js-yaml": "^4.1.0",
    "remark-directive": "^3.0.0",
    "remark-frontmatter": "^5.0.0",
    "remark-parse": "^11.0.0",
    "unified": "^11.0.5",
    "unist-util-visit": "^5.0.0"
  },
  "devDependencies": {
    "@types/better-sqlite3": "^7.6.10",
    "@types/cors": "^2.8.17",
    "@types/express": "^4.17.21",
    "@types/js-yaml": "^4.0.9",
    "@types/node": "^20.12.7",
    "tsx": "^4.7.2",
    "typescript": "^5.4.5",
    "vitest": "^1.5.0"
  }
}
```

- [ ] **Step 2: Create `backend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist"]
}
```

- [ ] **Step 3: Create `backend/.env.example`**

```
# Override content directory path
# Default resolves to repo root /content from backend/src/lib/
# CONTENT_DIR=/absolute/path/to/content
PORT=3001
```

- [ ] **Step 4: Create `backend/src/app.ts`**

```typescript
import express from 'express';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, data: { status: 'ok' } });
});

const PORT = process.env.PORT ?? 3001;
app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});

export default app;
```

- [ ] **Step 5: Install backend dependencies**

Run: `cd backend && pnpm install`
Expected: packages installed, no errors

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "feat: scaffold backend package"
```

---

### Task 2: Backend content lib + curriculum route

**Files:**
- Create: `content/curriculum.json` (minimal — s1/m3 with 2 lessons)
- Create: `content/s1-math-foundations/m3-information-theory/l1-entropy.md` (stub)
- Create: `backend/src/lib/content.ts`
- Create: `backend/src/routes/curriculum.ts`
- Create: `backend/src/tests/content.test.ts`
- Modify: `backend/src/app.ts`

- [ ] **Step 1: Write failing test**

Create `backend/src/tests/content.test.ts`:
```typescript
import { describe, it, expect } from 'vitest';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadCurriculum } from '../lib/content.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONTENT_DIR = path.resolve(__dirname, '../../../content');

describe('loadCurriculum', () => {
  it('returns a semesters array', async () => {
    const c = await loadCurriculum(CONTENT_DIR);
    expect(Array.isArray(c.semesters)).toBe(true);
    expect(c.semesters.length).toBeGreaterThan(0);
  });

  it('first semester has correct id', async () => {
    const c = await loadCurriculum(CONTENT_DIR);
    expect(c.semesters[0].id).toBe('s1-math-foundations');
  });
});
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd backend && pnpm test`
Expected: FAIL — module not found

- [ ] **Step 3: Create `content/curriculum.json`**

```json
{
  "semesters": [
    {
      "id": "s1-math-foundations",
      "title": "Semester 1: Mathematical Foundations",
      "description": "The mathematical language underpinning all modern ML.",
      "status": "available",
      "modules": [
        {
          "id": "m3-information-theory",
          "title": "Information Theory for ML",
          "description": "Entropy, KL divergence, mutual information, and connections to modern ML.",
          "status": "available",
          "lessons": [
            {
              "id": "l1-entropy",
              "title": "Shannon Entropy & Information Content",
              "estimatedMinutes": 25,
              "status": "available"
            },
            {
              "id": "l2-kl-divergence",
              "title": "KL Divergence & f-Divergences",
              "estimatedMinutes": 30,
              "status": "available"
            }
          ]
        }
      ]
    }
  ]
}
```

- [ ] **Step 4: Create stub lesson `content/s1-math-foundations/m3-information-theory/l1-entropy.md`**

```markdown
---
title: "Shannon Entropy & Information Content"
estimatedMinutes: 25
tags: ["information-theory", "entropy", "shannon"]
prerequisites: []
---

## Overview

Shannon entropy measures the average information content of a random variable.

$$H(X) = -\sum_{x} p(x) \log_2 p(x)$$

This lesson is a stub. Full content coming in Phase 5.

:::quiz
question: "What is the entropy of a fair coin flip?"
options:
  - "0 bits"
  - "1 bit"
  - "2 bits"
  - "0.5 bits"
correct: 1
explanation: "A fair coin flip has maximum entropy of 1 bit — complete uncertainty between two equally likely outcomes."
:::

The uniform distribution maximizes entropy for a fixed number of outcomes.
```

- [ ] **Step 5: Create `backend/src/lib/content.ts`**

```typescript
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const CONTENT_DIR = process.env.CONTENT_DIR
  ?? path.resolve(__dirname, '../../../content');

export interface LessonMeta {
  id: string;
  title: string;
  estimatedMinutes: number;
  status: 'available' | 'coming-soon';
}

export interface ModuleMeta {
  id: string;
  title: string;
  description?: string;
  status: 'available' | 'coming-soon';
  releaseNote?: string;
  lessons: LessonMeta[];
}

export interface SemesterMeta {
  id: string;
  title: string;
  description?: string;
  status: 'available' | 'coming-soon';
  modules: ModuleMeta[];
}

export interface Curriculum {
  semesters: SemesterMeta[];
}

export async function loadCurriculum(contentDir = CONTENT_DIR): Promise<Curriculum> {
  const raw = await fs.readFile(path.join(contentDir, 'curriculum.json'), 'utf-8');
  return JSON.parse(raw) as Curriculum;
}

export async function readLessonRaw(
  contentDir: string,
  semId: string,
  modId: string,
  lessonId: string
): Promise<string> {
  const filePath = path.join(contentDir, semId, modId, `${lessonId}.md`);
  return fs.readFile(filePath, 'utf-8');
}
```

- [ ] **Step 6: Run test — verify it passes**

Run: `cd backend && pnpm test`
Expected: PASS (2 tests)

- [ ] **Step 7: Create `backend/src/routes/curriculum.ts`**

```typescript
import { Router } from 'express';
import { loadCurriculum, CONTENT_DIR } from '../lib/content.js';

const router = Router();

router.get('/', async (_req, res) => {
  try {
    const curriculum = await loadCurriculum(CONTENT_DIR);
    res.json({ ok: true, data: curriculum });
  } catch (err) {
    console.error('Failed to load curriculum:', err);
    res.status(500).json({
      ok: false,
      error: { code: 'INTERNAL_ERROR', message: 'Failed to load curriculum' }
    });
  }
});

export default router;
```

- [ ] **Step 8: Update `backend/src/app.ts` to mount curriculum route**

```typescript
import express from 'express';
import cors from 'cors';
import curriculumRouter from './routes/curriculum.js';

const app = express();
app.use(cors());
app.use(express.json());

app.use('/api/curriculum', curriculumRouter);

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, data: { status: 'ok' } });
});

const PORT = process.env.PORT ?? 3001;
app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});

export default app;
```

- [ ] **Step 9: Commit**

```bash
git add backend/src/ content/
git commit -m "feat: add curriculum route and content loader"
```

---

### Task 3: Backend quiz parser + lessons route

**Files:**
- Create: `backend/src/lib/quiz.ts`
- Create: `backend/src/tests/quiz.test.ts`
- Create: `backend/src/routes/lessons.ts`
- Modify: `backend/src/app.ts`

- [ ] **Step 1: Write failing test for quiz parser**

Create `backend/src/tests/quiz.test.ts`:
```typescript
import { describe, it, expect } from 'vitest';
import { splitContent } from '../lib/quiz.js';

const SAMPLE_MD = `## Overview

Some prose here with math.

:::quiz
question: "What maximizes entropy?"
options:
  - "Uniform distribution"
  - "Skewed distribution"
correct: 0
explanation: "The uniform distribution maximizes entropy."
:::

More prose after the quiz.`;

describe('splitContent', () => {
  it('splits markdown into prose and quiz blocks', () => {
    const blocks = splitContent(SAMPLE_MD);
    expect(blocks.length).toBe(3);
    expect(blocks[0].type).toBe('markdown');
    expect(blocks[1].type).toBe('quiz');
    expect(blocks[2].type).toBe('markdown');
  });

  it('quiz block has correct fields', () => {
    const blocks = splitContent(SAMPLE_MD);
    const quiz = blocks[1];
    if (quiz.type !== 'quiz') throw new Error('expected quiz block');
    expect(quiz.question).toBe('What maximizes entropy?');
    expect(quiz.options).toHaveLength(2);
    expect(quiz.correct).toBe(0);
    expect(quiz.explanation).toBeTruthy();
  });

  it('returns single markdown block if no quiz directives', () => {
    const blocks = splitContent('Just prose here.');
    expect(blocks).toHaveLength(1);
    expect(blocks[0].type).toBe('markdown');
  });

  it('omits quiz blocks with missing required fields', () => {
    const md = `Prose.\n\n:::quiz\nquestion: "Missing correct field"\noptions:\n  - "A"\n:::\n\nMore.`;
    const blocks = splitContent(md);
    expect(blocks.every(b => b.type === 'markdown')).toBe(true);
  });
});
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd backend && pnpm test`
Expected: FAIL — module not found

- [ ] **Step 3: Create `backend/src/lib/quiz.ts`**

Uses `unified` + `remark-parse` + `remark-directive` to parse the AST, then walks it with `unist-util-visit` to find `containerDirective` nodes named `"quiz"`. Position offsets reconstruct interleaved markdown blocks from the original source string.

```typescript
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkDirective from 'remark-directive';
import remarkFrontmatter from 'remark-frontmatter';
import { visit } from 'unist-util-visit';
import yaml from 'js-yaml';
import type { Node } from 'unist';

export interface MarkdownBlock {
  type: 'markdown';
  raw: string;
}

export interface QuizBlock {
  type: 'quiz';
  question: string;
  options: string[];
  correct: number;
  explanation: string;
}

export type ContentBlock = MarkdownBlock | QuizBlock;

interface RawQuizFields {
  question: unknown;
  options: unknown;
  correct: unknown;
  explanation: unknown;
}

// Position data on remark nodes
interface NodeWithPosition extends Node {
  name?: string;
  position?: {
    start: { offset?: number };
    end: { offset?: number };
  };
}

function isValidQuiz(
  parsed: RawQuizFields
): parsed is { question: string; options: string[]; correct: number; explanation: string } {
  return (
    typeof parsed.question === 'string' &&
    Array.isArray(parsed.options) &&
    (parsed.options as unknown[]).length >= 2 &&
    typeof parsed.correct === 'number' &&
    typeof parsed.explanation === 'string'
  );
}

// The processor is reused across calls — unified processors are safe to reuse
// (each .parse() call returns a new tree; the processor is not mutated).
const processor = unified()
  .use(remarkParse)
  .use(remarkFrontmatter)
  .use(remarkDirective);

export function splitContent(markdown: string): ContentBlock[] {
  const tree = processor.parse(markdown);

  // Collect quiz directive positions in document order.
  // We use source-position offsets to slice YAML directly from the raw markdown string.
  // This avoids deep AST text traversal (remark parses YAML list values as listItem nodes,
  // not flat text children — AST text extraction would silently drop option values).
  const quizPositions: Array<{ start: number; end: number }> = [];

  visit(tree, 'containerDirective', (node: Node) => {
    const n = node as NodeWithPosition;
    if (n.name === 'quiz' && n.position) {
      quizPositions.push({
        start: n.position.start.offset ?? 0,
        end: n.position.end.offset ?? 0,
      });
    }
  });

  const blocks: ContentBlock[] = [];
  let lastIndex = 0;

  for (const { start, end } of quizPositions) {
    const preceding = markdown.slice(lastIndex, start).trim();
    if (preceding) {
      blocks.push({ type: 'markdown', raw: preceding });
    }

    // Extract YAML body by slicing between ":::quiz\n" and "\n:::"
    // Using the source string directly is more reliable than walking the AST.
    const blockSource = markdown.slice(start, end);
    const bodyMatch = blockSource.match(/^:::quiz\n([\s\S]*?)\n:::$/);
    const yamlText = bodyMatch ? bodyMatch[1] : '';

    try {
      const parsed = yaml.load(yamlText) as RawQuizFields;
      if (isValidQuiz(parsed)) {
        blocks.push({
          type: 'quiz',
          question: parsed.question,
          options: parsed.options,
          correct: parsed.correct,
          explanation: parsed.explanation,
        });
      } else {
        console.warn('Quiz block missing required fields — skipping');
      }
    } catch (err) {
      console.warn('Failed to parse quiz YAML — skipping:', err);
    }

    lastIndex = end;
  }

  const remaining = markdown.slice(lastIndex).trim();
  if (remaining) {
    blocks.push({ type: 'markdown', raw: remaining });
  }

  return blocks;
}
```

- [ ] **Step 4: Run test — verify it passes**

Run: `cd backend && pnpm test`
Expected: PASS (all tests)

- [ ] **Step 5: Create `backend/src/routes/lessons.ts`**

```typescript
import { Router } from 'express';
import yaml from 'js-yaml';
import { CONTENT_DIR, loadCurriculum, readLessonRaw } from '../lib/content.js';
import { splitContent } from '../lib/quiz.js';

const router = Router();

function parseFrontmatter(raw: string): {
  frontmatter: Record<string, unknown>;
  body: string;
} {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) return { frontmatter: {}, body: raw };
  try {
    const frontmatter = yaml.load(match[1]) as Record<string, unknown>;
    return { frontmatter, body: match[2] };
  } catch {
    return { frontmatter: {}, body: raw };
  }
}

async function getPrevNext(
  semId: string,
  modId: string,
  lessonId: string
): Promise<{
  prev: { semId: string; modId: string; lessonId: string; title: string } | null;
  next: { semId: string; modId: string; lessonId: string; title: string } | null;
}> {
  const curriculum = await loadCurriculum(CONTENT_DIR);
  const sem = curriculum.semesters.find(s => s.id === semId);
  if (!sem) return { prev: null, next: null };
  const mod = sem.modules.find(m => m.id === modId);
  if (!mod) return { prev: null, next: null };

  const idx = mod.lessons.findIndex(l => l.id === lessonId);
  if (idx === -1) return { prev: null, next: null };

  const prevLesson = idx > 0 ? mod.lessons[idx - 1] : null;
  const nextLesson = idx < mod.lessons.length - 1 ? mod.lessons[idx + 1] : null;

  return {
    prev: prevLesson ? { semId, modId, lessonId: prevLesson.id, title: prevLesson.title } : null,
    next: nextLesson ? { semId, modId, lessonId: nextLesson.id, title: nextLesson.title } : null,
  };
}

router.get('/:semId/:modId/:lessonId', async (req, res) => {
  const { semId, modId, lessonId } = req.params;
  try {
    const raw = await readLessonRaw(CONTENT_DIR, semId, modId, lessonId);
    const { frontmatter, body } = parseFrontmatter(raw);
    const content = splitContent(body);
    const { prev, next } = await getPrevNext(semId, modId, lessonId);

    res.json({
      ok: true,
      data: {
        id: lessonId,
        title: frontmatter.title ?? lessonId,
        semesterId: semId,
        moduleId: modId,
        estimatedMinutes: frontmatter.estimatedMinutes ?? null,
        tags: frontmatter.tags ?? [],
        prerequisites: frontmatter.prerequisites ?? [],
        content,
        prev,
        next,
      },
    });
  } catch (err: unknown) {
    const isNotFound =
      err !== null &&
      typeof err === 'object' &&
      'code' in err &&
      (err as { code: string }).code === 'ENOENT';

    if (isNotFound) {
      res.status(404).json({
        ok: false,
        error: { code: 'NOT_FOUND', message: 'Lesson not found' },
      });
    } else {
      console.error('Lesson route error:', err);
      res.status(500).json({
        ok: false,
        error: { code: 'INTERNAL_ERROR', message: 'Failed to load lesson' },
      });
    }
  }
});

export default router;
```

- [ ] **Step 6: Update `backend/src/app.ts` to mount lessons route**

```typescript
import express from 'express';
import cors from 'cors';
import curriculumRouter from './routes/curriculum.js';
import lessonsRouter from './routes/lessons.js';

const app = express();
app.use(cors());
app.use(express.json());

app.use('/api/curriculum', curriculumRouter);
app.use('/api/lessons', lessonsRouter);

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, data: { status: 'ok' } });
});

const PORT = process.env.PORT ?? 3001;
app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});

export default app;
```

- [ ] **Step 7: Commit**

```bash
git add backend/src/ content/
git commit -m "feat: add quiz parser and lessons route"
```

---

### Task 4: Backend SQLite + progress routes

**Files:**
- Create: `backend/src/lib/db.ts`
- Create: `backend/src/routes/progress.ts`
- Create: `backend/src/tests/progress.test.ts`
- Create: `backend/data/.gitkeep`
- Modify: `backend/src/app.ts`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing test for db**

Create `backend/src/tests/progress.test.ts`:
```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import Database from 'better-sqlite3';
import { initDb, getProgress, upsertLessonStatus, saveQuizAnswer } from '../lib/db.js';

let db: Database.Database;

beforeEach(() => {
  db = new Database(':memory:');
  initDb(db);
});

describe('progress db', () => {
  it('returns empty progress for new session', () => {
    const progress = getProgress(db, 'session-abc');
    expect(progress.lessons).toEqual({});
    expect(progress.stats.totalCompleted).toBe(0);
  });

  it('upserts lesson status to completed', () => {
    upsertLessonStatus(db, 'session-abc', 'l1-entropy', 'completed');
    const progress = getProgress(db, 'session-abc');
    expect(progress.lessons['l1-entropy'].status).toBe('completed');
    expect(progress.lessons['l1-entropy'].completedAt).not.toBeNull();
  });

  it('saves quiz answer without changing lesson status to completed', () => {
    upsertLessonStatus(db, 'session-abc', 'l1-entropy', 'in_progress');
    saveQuizAnswer(db, 'session-abc', 'l1-entropy', 0, 1);
    const progress = getProgress(db, 'session-abc');
    expect(progress.lessons['l1-entropy'].status).toBe('in_progress');
    expect(progress.lessons['l1-entropy'].quizAnswers).toEqual({ '0': 1 });
  });

  it('calculates totalCompleted correctly', () => {
    upsertLessonStatus(db, 'session-abc', 'l1-entropy', 'completed');
    upsertLessonStatus(db, 'session-abc', 'l2-kl-divergence', 'completed');
    const progress = getProgress(db, 'session-abc');
    expect(progress.stats.totalCompleted).toBe(2);
  });
});
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd backend && pnpm test`
Expected: FAIL — module not found

- [ ] **Step 3: Create `backend/src/lib/db.ts`**

```typescript
import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../../data');

export function initDb(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS progress (
      session_id   TEXT NOT NULL,
      lesson_id    TEXT NOT NULL,
      status       TEXT NOT NULL DEFAULT 'not_started',
      quiz_answers TEXT,
      started_at   TEXT,
      completed_at TEXT,
      updated_at   TEXT NOT NULL,
      PRIMARY KEY (session_id, lesson_id)
    );

    CREATE TABLE IF NOT EXISTS sessions (
      session_id      TEXT PRIMARY KEY,
      first_seen_at   TEXT NOT NULL,
      last_active_at  TEXT NOT NULL,
      streak_days     INTEGER NOT NULL DEFAULT 0,
      last_streak_at  TEXT
    );
  `);
}

export function openDb(): Database.Database {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  const dbPath = path.join(DATA_DIR, 'progress.db');
  const db = new Database(dbPath);
  initDb(db);
  return db;
}

function ensureSession(db: Database.Database, sessionId: string): void {
  const now = new Date().toISOString();
  const existing = db
    .prepare('SELECT session_id, last_streak_at, streak_days FROM sessions WHERE session_id = ?')
    .get(sessionId) as
    | { session_id: string; last_streak_at: string | null; streak_days: number }
    | undefined;

  if (!existing) {
    db.prepare(`
      INSERT INTO sessions (session_id, first_seen_at, last_active_at, streak_days, last_streak_at)
      VALUES (?, ?, ?, 1, ?)
    `).run(sessionId, now, now, now);
    return;
  }

  const lastDate = existing.last_streak_at
    ? new Date(existing.last_streak_at).toDateString()
    : null;
  const todayStr = new Date().toDateString();
  let streak = existing.streak_days;

  if (lastDate !== todayStr) {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    streak = lastDate === yesterday.toDateString() ? streak + 1 : 1;
    db.prepare(`
      UPDATE sessions SET last_active_at = ?, streak_days = ?, last_streak_at = ?
      WHERE session_id = ?
    `).run(now, streak, now, sessionId);
  } else {
    db.prepare('UPDATE sessions SET last_active_at = ? WHERE session_id = ?').run(
      now,
      sessionId
    );
  }
}

export interface LessonProgress {
  status: 'not_started' | 'in_progress' | 'completed';
  completedAt: string | null;
  quizAnswers: Record<string, number>;
}

export interface ProgressData {
  lessons: Record<string, LessonProgress>;
  stats: {
    streak: number;
    totalCompleted: number;
    lastActivityAt: string | null;
  };
}

export function getProgress(db: Database.Database, sessionId: string): ProgressData {
  const rows = db
    .prepare(
      'SELECT lesson_id, status, completed_at, quiz_answers FROM progress WHERE session_id = ?'
    )
    .all(sessionId) as Array<{
    lesson_id: string;
    status: string;
    completed_at: string | null;
    quiz_answers: string | null;
  }>;

  const lessons: Record<string, LessonProgress> = {};
  for (const row of rows) {
    lessons[row.lesson_id] = {
      status: row.status as LessonProgress['status'],
      completedAt: row.completed_at,
      quizAnswers: row.quiz_answers ? (JSON.parse(row.quiz_answers) as Record<string, number>) : {},
    };
  }

  const session = db
    .prepare('SELECT streak_days, last_active_at FROM sessions WHERE session_id = ?')
    .get(sessionId) as { streak_days: number; last_active_at: string } | undefined;

  const totalCompleted = Object.values(lessons).filter(l => l.status === 'completed').length;

  return {
    lessons,
    stats: {
      streak: session?.streak_days ?? 0,
      totalCompleted,
      lastActivityAt: session?.last_active_at ?? null,
    },
  };
}

export function upsertLessonStatus(
  db: Database.Database,
  sessionId: string,
  lessonId: string,
  status: 'in_progress' | 'completed'
): void {
  const now = new Date().toISOString();
  ensureSession(db, sessionId);

  const existing = db
    .prepare(
      'SELECT status, quiz_answers, started_at FROM progress WHERE session_id = ? AND lesson_id = ?'
    )
    .get(sessionId, lessonId) as
    | { status: string; quiz_answers: string | null; started_at: string | null }
    | undefined;

  if (!existing) {
    db.prepare(`
      INSERT INTO progress (session_id, lesson_id, status, started_at, completed_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(sessionId, lessonId, status, now, status === 'completed' ? now : null, now);
  } else {
    db.prepare(`
      UPDATE progress SET status = ?, completed_at = ?, updated_at = ?
      WHERE session_id = ? AND lesson_id = ?
    `).run(
      status,
      status === 'completed' ? now : null,
      now,
      sessionId,
      lessonId
    );
  }
}

export function saveQuizAnswer(
  db: Database.Database,
  sessionId: string,
  lessonId: string,
  quizIndex: number,
  selectedOption: number
): void {
  const now = new Date().toISOString();
  ensureSession(db, sessionId);

  const existing = db
    .prepare(
      'SELECT quiz_answers, status FROM progress WHERE session_id = ? AND lesson_id = ?'
    )
    .get(sessionId, lessonId) as { quiz_answers: string | null; status: string } | undefined;

  const answers: Record<string, number> = existing?.quiz_answers
    ? (JSON.parse(existing.quiz_answers) as Record<string, number>)
    : {};
  answers[String(quizIndex)] = selectedOption;
  const answersJson = JSON.stringify(answers);

  if (!existing) {
    db.prepare(`
      INSERT INTO progress (session_id, lesson_id, status, quiz_answers, started_at, updated_at)
      VALUES (?, ?, 'in_progress', ?, ?, ?)
    `).run(sessionId, lessonId, answersJson, now, now);
  } else {
    const newStatus = existing.status === 'not_started' ? 'in_progress' : existing.status;
    db.prepare(`
      UPDATE progress SET quiz_answers = ?, status = ?, updated_at = ?
      WHERE session_id = ? AND lesson_id = ?
    `).run(answersJson, newStatus, now, sessionId, lessonId);
  }
}
```

- [ ] **Step 4: Run test — verify it passes**

Run: `cd backend && pnpm test`
Expected: PASS (all tests)

- [ ] **Step 5: Create `backend/src/routes/progress.ts`**

```typescript
import { Router, Request, Response } from 'express';
import { openDb, getProgress, upsertLessonStatus, saveQuizAnswer } from '../lib/db.js';

const router = Router();
const db = openDb();

function getSessionId(req: Request): string | null {
  const id = req.headers['x-session-id'];
  if (typeof id !== 'string' || !id.trim()) return null;
  return id.trim();
}

router.get('/', (req: Request, res: Response) => {
  const sessionId = getSessionId(req);
  if (!sessionId) {
    return res.status(400).json({
      ok: false,
      error: { code: 'BAD_REQUEST', message: 'X-Session-Id header required' },
    });
  }
  res.json({ ok: true, data: getProgress(db, sessionId) });
});

router.post('/:lessonId', (req: Request, res: Response) => {
  const sessionId = getSessionId(req);
  if (!sessionId) {
    return res.status(400).json({
      ok: false,
      error: { code: 'BAD_REQUEST', message: 'X-Session-Id header required' },
    });
  }

  const { lessonId } = req.params;
  const { status, quizAnswer } = req.body as {
    status?: string;
    quizAnswer?: { quizIndex: number; selectedOption: number };
  };

  if (status !== undefined && quizAnswer !== undefined) {
    return res.status(400).json({
      ok: false,
      error: { code: 'BAD_REQUEST', message: 'Provide either status or quizAnswer, not both' },
    });
  }

  if (status !== undefined) {
    if (status !== 'completed' && status !== 'in_progress') {
      return res.status(400).json({
        ok: false,
        error: { code: 'BAD_REQUEST', message: 'status must be "completed" or "in_progress"' },
      });
    }
    upsertLessonStatus(db, sessionId, lessonId, status);
    return res.json({ ok: true, data: null });
  }

  if (quizAnswer !== undefined) {
    const { quizIndex, selectedOption } = quizAnswer;
    if (typeof quizIndex !== 'number' || typeof selectedOption !== 'number') {
      return res.status(400).json({
        ok: false,
        error: { code: 'BAD_REQUEST', message: 'quizAnswer needs numeric quizIndex and selectedOption' },
      });
    }
    saveQuizAnswer(db, sessionId, lessonId, quizIndex, selectedOption);
    return res.json({ ok: true, data: null });
  }

  res.status(400).json({
    ok: false,
    error: { code: 'BAD_REQUEST', message: 'Body must contain status or quizAnswer' },
  });
});

export default router;
```

- [ ] **Step 6: Update `backend/src/app.ts` with all three routes**

```typescript
import express from 'express';
import cors from 'cors';
import curriculumRouter from './routes/curriculum.js';
import lessonsRouter from './routes/lessons.js';
import progressRouter from './routes/progress.js';

const app = express();
app.use(cors());
app.use(express.json());

app.use('/api/curriculum', curriculumRouter);
app.use('/api/lessons', lessonsRouter);
app.use('/api/progress', progressRouter);

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, data: { status: 'ok' } });
});

const PORT = process.env.PORT ?? 3001;
app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});

export default app;
```

- [ ] **Step 7: Add data directory to .gitignore, create placeholder**

Append to `.gitignore`:
```
backend/data/*.db
```

Run: `touch backend/data/.gitkeep`

- [ ] **Step 8: Run all tests**

Run: `cd backend && pnpm test`
Expected: PASS (all 8+ tests)

- [ ] **Step 9: Commit**

```bash
git add backend/src/ backend/data/.gitkeep .gitignore
git commit -m "feat: add SQLite progress store and progress routes"
```

---

## Chunk 2: Frontend Foundation

### Task 5: Scaffold frontend package

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/index.css`
- Create: `frontend/src/main.tsx`

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "@tanstack/react-query": "^5.29.2",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-markdown": "^9.0.1",
    "react-router-dom": "^6.22.3",
    "react-syntax-highlighter": "^15.5.0",
    "rehype-katex": "^7.0.1",
    "remark-directive": "^3.0.0",
    "remark-math": "^6.0.0",
    "katex": "^0.16.10"
  },
  "devDependencies": {
    "@types/katex": "^0.16.7",
    "@types/react": "^18.2.74",
    "@types/react-dom": "^18.2.23",
    "@types/react-syntax-highlighter": "^15.5.13",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.3",
    "typescript": "^5.4.5",
    "vite": "^5.2.8",
    "vitest": "^1.5.0"
  }
}
```

- [ ] **Step 2: Create `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

- [ ] **Step 3: Create `frontend/tsconfig.node.json`**

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 4: Create `frontend/vite.config.ts`**

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:3001', changeOrigin: true },
    },
  },
});
```

- [ ] **Step 5: Create `frontend/tailwind.config.js`**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'surface-deep': '#0a0f1e',
      },
    },
  },
  plugins: [],
};
```

- [ ] **Step 6: Create `frontend/postcss.config.js`**

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

- [ ] **Step 7: Create `frontend/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css"
    />
    <title>AI Mastery Hub</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 8: Create `frontend/src/index.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  html, body, #root {
    @apply h-full;
  }
  body {
    @apply bg-slate-900 text-slate-100;
  }
}

.katex-display {
  @apply my-6 overflow-x-auto;
}
```

- [ ] **Step 9: Create `frontend/src/main.tsx`**

```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App.tsx';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
```

- [ ] **Step 10: Install frontend dependencies**

Run: `cd frontend && pnpm install`
Expected: packages installed, no errors

- [ ] **Step 11: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold frontend package with Vite + Tailwind"
```

---

### Task 6: Frontend lib — api wrapper, session, shared types

**Files:**
- Create: `frontend/src/types.ts`
- Create: `frontend/src/lib/session.ts`
- Create: `frontend/src/lib/api.ts`

- [ ] **Step 1: Create `frontend/src/types.ts`**

```typescript
export interface LessonMeta {
  id: string;
  title: string;
  estimatedMinutes: number;
  status: 'available' | 'coming-soon';
}

export interface ModuleMeta {
  id: string;
  title: string;
  description?: string;
  status: 'available' | 'coming-soon';
  releaseNote?: string;
  lessons: LessonMeta[];
}

export interface SemesterMeta {
  id: string;
  title: string;
  description?: string;
  status: 'available' | 'coming-soon';
  modules: ModuleMeta[];
}

export interface Curriculum {
  semesters: SemesterMeta[];
}

export interface MarkdownBlock {
  type: 'markdown';
  raw: string;
}

export interface QuizBlock {
  type: 'quiz';
  question: string;
  options: string[];
  correct: number;
  explanation: string;
}

export type ContentBlock = MarkdownBlock | QuizBlock;

export interface LessonNavRef {
  semId: string;
  modId: string;
  lessonId: string;
  title: string;
}

export interface Lesson {
  id: string;
  title: string;
  semesterId: string;
  moduleId: string;
  estimatedMinutes: number | null;
  tags: string[];
  prerequisites: string[];
  content: ContentBlock[];
  prev: LessonNavRef | null;
  next: LessonNavRef | null;
}

export interface LessonProgress {
  status: 'not_started' | 'in_progress' | 'completed';
  completedAt: string | null;
  quizAnswers: Record<string, number>;
}

export interface ProgressData {
  lessons: Record<string, LessonProgress>;
  stats: {
    streak: number;
    totalCompleted: number;
    lastActivityAt: string | null;
  };
}
```

- [ ] **Step 2: Create `frontend/src/lib/session.ts`**

```typescript
const SESSION_KEY = 'amh_session_id';
const FIRST_VISIT_KEY = 'amh_first_visit';

function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export function getSessionId(): string {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = generateUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

export function isFirstVisit(): boolean {
  return localStorage.getItem(FIRST_VISIT_KEY) === null;
}

export function markVisited(): void {
  localStorage.setItem(FIRST_VISIT_KEY, 'true');
}
```

- [ ] **Step 3: Create `frontend/src/lib/api.ts`**

```typescript
import { getSessionId } from './session.ts';

type ApiResponse<T> =
  | { ok: true; data: T }
  | { ok: false; error: { code: string; message: string } };

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-Session-Id': getSessionId(),
      ...options?.headers,
    },
  });

  const json = (await res.json()) as ApiResponse<T>;
  if (!json.ok) {
    throw new Error(json.error.message);
  }
  return json.data;
}

export const api = {
  get: <T>(path: string) => apiFetch<T>(path),
  post: <T>(path: string, body: unknown) =>
    apiFetch<T>(path, { method: 'POST', body: JSON.stringify(body) }),
};
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/
git commit -m "feat: add api wrapper, session management, and shared types"
```

---

### Task 7: App shell — AppShell, Header, Sidebar, hooks

**Files:**
- Create: `frontend/src/hooks/useCurriculum.ts`
- Create: `frontend/src/hooks/useProgress.ts`
- Create: `frontend/src/components/layout/Header.tsx`
- Create: `frontend/src/components/layout/Sidebar.tsx`
- Create: `frontend/src/components/layout/AppShell.tsx`
- Create: `frontend/src/components/ErrorBoundary.tsx`
- Create: `frontend/src/components/Toast.tsx`
- Create: `frontend/src/pages/DashboardPage.tsx` (stub)
- Create: `frontend/src/pages/SemesterPage.tsx` (stub)
- Create: `frontend/src/pages/LessonPage.tsx` (stub)
- Create: `frontend/src/App.tsx`

- [ ] **Step 1: Create `frontend/src/hooks/useCurriculum.ts`**

```typescript
import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api.ts';
import type { Curriculum } from '../types.ts';

export function useCurriculum() {
  return useQuery({
    queryKey: ['curriculum'],
    queryFn: () => api.get<Curriculum>('/api/curriculum'),
    staleTime: Infinity,
  });
}
```

- [ ] **Step 2: Create `frontend/src/hooks/useProgress.ts`**

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api.ts';
import type { ProgressData } from '../types.ts';

export function useProgress() {
  return useQuery({
    queryKey: ['progress'],
    queryFn: () => api.get<ProgressData>('/api/progress'),
  });
}

export function useMarkComplete() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (lessonId: string) =>
      api.post<null>(`/api/progress/${lessonId}`, { status: 'completed' }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['progress'] });
    },
  });
}

export function useSaveQuizAnswer() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      lessonId,
      quizIndex,
      selectedOption,
    }: {
      lessonId: string;
      quizIndex: number;
      selectedOption: number;
    }) =>
      api.post<null>(`/api/progress/${lessonId}`, {
        quizAnswer: { quizIndex, selectedOption },
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['progress'] });
    },
  });
}
```

- [ ] **Step 3: Create `frontend/src/components/layout/Header.tsx`**

```typescript
import { Link } from 'react-router-dom';

export function Header() {
  return (
    <header className="h-12 bg-slate-900 border-b border-white/5 flex items-center px-6 flex-shrink-0 z-10">
      <Link to="/" className="flex items-center gap-2">
        <div className="w-5 h-5 rounded bg-gradient-to-br from-indigo-500 to-violet-500 flex-shrink-0" />
        <span className="font-extrabold tracking-tight text-slate-100 text-sm">
          AI Mastery Hub
        </span>
      </Link>
    </header>
  );
}
```

- [ ] **Step 4: Create `frontend/src/components/layout/Sidebar.tsx`**

```typescript
import { Link, useParams } from 'react-router-dom';
import { useCurriculum } from '../../hooks/useCurriculum.ts';
import type { SemesterMeta, ModuleMeta, LessonMeta } from '../../types.ts';

function LessonItem({
  lesson,
  semId,
  modId,
  isActive,
  isCompleted,
}: {
  lesson: LessonMeta;
  semId: string;
  modId: string;
  isActive: boolean;
  isCompleted: boolean;
}) {
  const base = 'flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-colors pl-8';

  if (lesson.status === 'coming-soon') {
    return (
      <div className={`${base} text-slate-600 cursor-default`}>
        <span className="w-3 h-3 flex-shrink-0" />
        {lesson.title}
      </div>
    );
  }

  return (
    <Link
      to={`/lesson/${semId}/${modId}/${lesson.id}`}
      className={`${base} ${
        isActive
          ? 'bg-indigo-500/20 text-indigo-300 font-medium'
          : isCompleted
          ? 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
          : 'text-slate-300 hover:text-slate-100 hover:bg-white/5'
      }`}
    >
      {isCompleted ? (
        <span className="w-3 text-emerald-500 flex-shrink-0">✓</span>
      ) : isActive ? (
        <span className="w-3 h-3 rounded-full bg-indigo-500 flex-shrink-0 inline-block" />
      ) : (
        <span className="w-3 h-3 flex-shrink-0" />
      )}
      {lesson.title}
    </Link>
  );
}

function ModuleSection({
  mod,
  semId,
  activeLessonId,
  completedLessons,
}: {
  mod: ModuleMeta;
  semId: string;
  activeLessonId?: string;
  completedLessons: Set<string>;
}) {
  if (mod.status !== 'available') {
    return (
      <div className="flex items-center justify-between px-3 py-1.5 mb-1">
        <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider truncate">
          {mod.title}
        </span>
        <span className="text-[10px] text-slate-700 bg-slate-800 px-1.5 py-0.5 rounded flex-shrink-0 ml-2">
          soon
        </span>
      </div>
    );
  }

  return (
    <div className="mb-2">
      <div className="px-3 py-1 text-xs font-semibold text-slate-400 uppercase tracking-wider">
        {mod.title}
      </div>
      {mod.lessons.map(lesson => (
        <LessonItem
          key={lesson.id}
          lesson={lesson}
          semId={semId}
          modId={mod.id}
          isActive={lesson.id === activeLessonId}
          isCompleted={completedLessons.has(lesson.id)}
        />
      ))}
    </div>
  );
}

function SemesterSection({
  sem,
  activeLessonId,
  completedLessons,
}: {
  sem: SemesterMeta;
  activeLessonId?: string;
  completedLessons: Set<string>;
}) {
  return (
    <div className="mb-5">
      <div className="px-3 py-2 flex items-center justify-between">
        <span
          className={`text-sm font-bold ${
            sem.status === 'available' ? 'text-slate-200' : 'text-slate-600'
          }`}
        >
          {sem.title}
        </span>
        {sem.status !== 'available' && (
          <span className="text-[10px] text-slate-700 bg-slate-800/80 px-1.5 py-0.5 rounded flex-shrink-0 ml-2">
            coming soon
          </span>
        )}
      </div>
      {sem.status === 'available' &&
        sem.modules.map(mod => (
          <ModuleSection
            key={mod.id}
            mod={mod}
            semId={sem.id}
            activeLessonId={activeLessonId}
            completedLessons={completedLessons}
          />
        ))}
    </div>
  );
}

export function Sidebar({ completedLessons = new Set<string>() }: { completedLessons?: Set<string> }) {
  const { data: curriculum, isLoading } = useCurriculum();
  const params = useParams<{ lessonId?: string }>();

  return (
    <aside className="w-[280px] flex-shrink-0 bg-slate-900 border-r border-white/5 overflow-y-auto">
      <div className="pt-4 pb-8">
        {isLoading && (
          <div className="px-4 text-slate-600 text-sm animate-pulse">Loading…</div>
        )}
        {curriculum?.semesters.map(sem => (
          <SemesterSection
            key={sem.id}
            sem={sem}
            activeLessonId={params.lessonId}
            completedLessons={completedLessons}
          />
        ))}
      </div>
    </aside>
  );
}
```

- [ ] **Step 5: Create `frontend/src/components/ErrorBoundary.tsx`**

```typescript
import { Component, ReactNode } from 'react';
import { Link } from 'react-router-dom';

interface Props { children: ReactNode }
interface State { hasError: boolean }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    console.error('ErrorBoundary:', error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="max-w-2xl mx-auto px-8 py-24 text-center">
          <h2 className="text-xl font-bold text-red-400 mb-3">Something went wrong</h2>
          <Link
            to="/"
            onClick={() => this.setState({ hasError: false })}
            className="text-indigo-400 hover:text-indigo-300"
          >
            ← Return to dashboard
          </Link>
        </div>
      );
    }
    return this.props.children;
  }
}
```

- [ ] **Step 6: Create `frontend/src/components/Toast.tsx`**

```typescript
import { useEffect, useState } from 'react';
import { isFirstVisit, markVisited } from '../lib/session.ts';

export function FirstVisitToast() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (isFirstVisit()) {
      markVisited();
      setVisible(true);
      const t = setTimeout(() => setVisible(false), 6000);
      return () => clearTimeout(t);
    }
  }, []);

  if (!visible) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 max-w-sm p-4 rounded-xl bg-slate-800 border border-indigo-500/30 shadow-2xl">
      <p className="text-sm font-semibold text-slate-200">Welcome to AI Mastery Hub</p>
      <p className="text-xs text-slate-400 mt-1">
        Your progress is saved locally in this browser. No account needed.
      </p>
    </div>
  );
}
```

- [ ] **Step 7: Create `frontend/src/components/layout/AppShell.tsx`**

```typescript
import { Outlet } from 'react-router-dom';
import { Header } from './Header.tsx';
import { Sidebar } from './Sidebar.tsx';
import { ErrorBoundary } from '../ErrorBoundary.tsx';
import { FirstVisitToast } from '../Toast.tsx';
import { useProgress } from '../../hooks/useProgress.ts';

export function AppShell() {
  const { data: progress } = useProgress();
  const completedLessons = new Set(
    Object.entries(progress?.lessons ?? {})
      .filter(([, lp]) => lp.status === 'completed')
      .map(([id]) => id)
  );

  return (
    <div className="flex flex-col h-screen">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar completedLessons={completedLessons} />
        <main className="flex-1 overflow-y-auto bg-surface-deep">
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
      <FirstVisitToast />
    </div>
  );
}
```

- [ ] **Step 8: Create stub pages**

`frontend/src/pages/DashboardPage.tsx`:
```typescript
export default function DashboardPage() {
  return (
    <div className="max-w-4xl mx-auto px-8 py-12">
      <h1 className="text-2xl font-extrabold tracking-tight">Dashboard</h1>
      <p className="text-slate-400 mt-2">Coming in Task 10.</p>
    </div>
  );
}
```

`frontend/src/pages/SemesterPage.tsx`:
```typescript
import { useParams } from 'react-router-dom';

export default function SemesterPage() {
  const { semId } = useParams();
  return (
    <div className="max-w-4xl mx-auto px-8 py-12">
      <p className="text-slate-500 text-sm font-mono">{semId}</p>
      <h1 className="text-2xl font-extrabold tracking-tight mt-1">Coming Soon</h1>
      <p className="text-slate-400 mt-3">This semester is under construction.</p>
    </div>
  );
}
```

`frontend/src/pages/LessonPage.tsx`:
```typescript
import { useParams } from 'react-router-dom';

export default function LessonPage() {
  const { semId, modId, lessonId } = useParams();
  return (
    <div className="max-w-3xl mx-auto px-8 py-12">
      <p className="text-slate-500 text-sm font-mono">{semId} / {modId} / {lessonId}</p>
      <h1 className="text-2xl font-extrabold tracking-tight mt-1">Lesson</h1>
      <p className="text-slate-400 mt-3">Lesson reader coming in Task 8.</p>
    </div>
  );
}
```

- [ ] **Step 9: Create `frontend/src/App.tsx`**

```typescript
import { Routes, Route, Navigate } from 'react-router-dom';
import { AppShell } from './components/layout/AppShell.tsx';
import DashboardPage from './pages/DashboardPage.tsx';
import SemesterPage from './pages/SemesterPage.tsx';
import LessonPage from './pages/LessonPage.tsx';

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<DashboardPage />} />
        <Route path="/semester/:semId" element={<SemesterPage />} />
        <Route path="/lesson/:semId/:modId/:lessonId" element={<LessonPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
```

- [ ] **Step 10: Verify app shell loads in browser**

Start both:
```bash
# Terminal 1
cd backend && pnpm dev
# Terminal 2
cd frontend && pnpm dev
```

Open http://localhost:5173
Expected:
- Dark background, "AI Mastery Hub" header
- Sidebar shows S1 with Information Theory module and 2 lesson links
- Clicking a lesson navigates to `/lesson/s1.../l1-entropy`
- No console errors

- [ ] **Step 11: Commit**

```bash
git add frontend/src/
git commit -m "feat: add app shell, sidebar, hooks, and stub pages"
```

---

## Chunk 3: Lesson Reader

### Task 8: LessonReader components

**Files:**
- Create: `frontend/src/hooks/useLesson.ts`
- Create: `frontend/src/components/reader/CodeBlock.tsx`
- Create: `frontend/src/components/reader/InlineQuiz.tsx`
- Create: `frontend/src/components/reader/LessonReader.tsx`

- [ ] **Step 1: Create `frontend/src/hooks/useLesson.ts`**

```typescript
import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api.ts';
import type { Lesson } from '../types.ts';

export function useLesson(semId: string, modId: string, lessonId: string) {
  return useQuery({
    queryKey: ['lesson', semId, modId, lessonId],
    queryFn: () => api.get<Lesson>(`/api/lessons/${semId}/${modId}/${lessonId}`),
    enabled: Boolean(semId && modId && lessonId),
  });
}
```

- [ ] **Step 2: Create `frontend/src/components/reader/CodeBlock.tsx`**

```typescript
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useState } from 'react';

interface Props {
  language?: string;
  children: string;
}

export function CodeBlock({ language = 'python', children }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    void navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-6 rounded-lg overflow-hidden border border-white/8">
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-white/8">
        <span className="text-xs text-slate-500 font-mono">{language}</span>
        <button
          onClick={handleCopy}
          className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        customStyle={{ margin: 0, padding: '1rem', background: 'rgb(15, 23, 42)', fontSize: '0.875rem' }}
        PreTag="div"
      >
        {children}
      </SyntaxHighlighter>
    </div>
  );
}
```

- [ ] **Step 3: Create `frontend/src/components/reader/InlineQuiz.tsx`**

```typescript
import { useState } from 'react';
import type { QuizBlock } from '../../types.ts';

interface Props {
  quizIndex: number;
  block: QuizBlock;
  savedAnswer?: number;
  onAnswer: (selectedOption: number) => void;
}

export function InlineQuiz({ block, savedAnswer, onAnswer }: Props) {
  const [selected, setSelected] = useState<number | undefined>(savedAnswer);
  const answered = selected !== undefined;
  const isCorrect = selected === block.correct;

  const handleSelect = (idx: number) => {
    if (answered) return;
    setSelected(idx);
    onAnswer(idx);
  };

  return (
    <div className="my-8 p-6 rounded-xl bg-slate-800/60 border border-white/8">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-5 h-5 rounded bg-indigo-500/20 flex items-center justify-center flex-shrink-0">
          <span className="text-indigo-400 text-xs font-bold">?</span>
        </div>
        <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
          Knowledge Check
        </span>
      </div>

      <p className="text-slate-200 font-medium mb-4">{block.question}</p>

      <div className="space-y-2">
        {block.options.map((option, idx) => {
          let style = 'border-white/10 text-slate-300 hover:border-indigo-500/50 hover:bg-indigo-500/5';
          if (answered) {
            if (idx === block.correct) style = 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300';
            else if (idx === selected) style = 'border-red-500/50 bg-red-500/10 text-red-300';
            else style = 'border-white/5 text-slate-600';
          }

          return (
            <button
              key={idx}
              onClick={() => handleSelect(idx)}
              disabled={answered}
              className={`w-full text-left px-4 py-3 rounded-lg border text-sm transition-all ${style} ${answered ? 'cursor-default' : 'cursor-pointer'}`}
            >
              <span className="font-mono text-xs mr-3 opacity-60">
                {String.fromCharCode(65 + idx)}.
              </span>
              {option}
            </button>
          );
        })}
      </div>

      {answered && (
        <div
          className={`mt-4 p-4 rounded-lg text-sm ${
            isCorrect
              ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-300'
              : 'bg-amber-500/10 border border-amber-500/20 text-amber-300'
          }`}
        >
          <span className="font-semibold mr-1">{isCorrect ? 'Correct!' : 'Not quite.'}</span>
          {block.explanation}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Create `frontend/src/components/reader/LessonReader.tsx`**

```typescript
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import type { Components } from 'react-markdown';
import { CodeBlock } from './CodeBlock.tsx';
import { InlineQuiz } from './InlineQuiz.tsx';
import type { ContentBlock, Lesson } from '../../types.ts';

const mdComponents: Components = {
  code({ className, children }) {
    const match = /language-(\w+)/.exec(className ?? '');
    const code = String(children).replace(/\n$/, '');
    if (match) return <CodeBlock language={match[1]}>{code}</CodeBlock>;
    return (
      <code className="bg-slate-800 text-indigo-300 px-1.5 py-0.5 rounded text-sm font-mono">
        {children}
      </code>
    );
  },
  h2({ children }) {
    return (
      <h2 className="text-xl font-extrabold tracking-tight text-slate-100 mt-10 mb-4 border-b border-white/5 pb-2">
        {children}
      </h2>
    );
  },
  h3({ children }) {
    return <h3 className="text-lg font-bold text-slate-200 mt-8 mb-3">{children}</h3>;
  },
  p({ children }) {
    return <p className="text-slate-300 leading-relaxed mb-4">{children}</p>;
  },
  ul({ children }) {
    return <ul className="list-disc pl-6 text-slate-300 mb-4 space-y-1">{children}</ul>;
  },
  ol({ children }) {
    return <ol className="list-decimal pl-6 text-slate-300 mb-4 space-y-1">{children}</ol>;
  },
  blockquote({ children }) {
    return (
      <blockquote className="border-l-4 border-indigo-500 pl-4 my-6 text-slate-300 bg-indigo-500/5 py-3 rounded-r-lg">
        {children}
      </blockquote>
    );
  },
};

interface Props {
  lesson: Lesson;
  quizAnswers: Record<string, number>;
  onQuizAnswer: (quizIndex: number, selectedOption: number) => void;
}

export function LessonReader({ lesson, quizAnswers, onQuizAnswer }: Props) {
  // quizCount tracks the index among quiz blocks only (0-based), NOT the full content array index.
  // This ensures quizAnswers keys ("0", "1", ...) always refer to the Nth quiz in the lesson,
  // regardless of how many markdown blocks precede it.
  let quizCount = 0;

  return (
    <div className="max-w-3xl">
      {lesson.content.map((block: ContentBlock, i: number) => {
        if (block.type === 'markdown') {
          return (
            <ReactMarkdown
              key={i}
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={mdComponents}
            >
              {block.raw}
            </ReactMarkdown>
          );
        }
        const qIdx = quizCount++;
        return (
          <InlineQuiz
            key={i}
            quizIndex={qIdx}
            block={block}
            savedAnswer={quizAnswers[String(qIdx)]}
            onAnswer={selected => onQuizAnswer(qIdx, selected)}
          />
        );
      })}
    </div>
  );
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/reader/ frontend/src/hooks/useLesson.ts
git commit -m "feat: add lesson reader with markdown, KaTeX, code, and inline quiz"
```

---

### Task 9: Wire LessonPage

**Files:**
- Modify: `frontend/src/pages/LessonPage.tsx`

- [ ] **Step 1: Replace `frontend/src/pages/LessonPage.tsx` with full implementation**

```typescript
import { useParams, Link } from 'react-router-dom';
import { useLesson } from '../hooks/useLesson.ts';
import { useProgress, useMarkComplete, useSaveQuizAnswer } from '../hooks/useProgress.ts';
import { LessonReader } from '../components/reader/LessonReader.tsx';

export default function LessonPage() {
  const { semId = '', modId = '', lessonId = '' } = useParams();
  const { data: lesson, isLoading, isError } = useLesson(semId, modId, lessonId);
  const { data: progress } = useProgress();
  const markComplete = useMarkComplete();
  const saveQuizAnswer = useSaveQuizAnswer();

  if (isLoading) {
    return (
      <div className="max-w-3xl mx-auto px-8 py-12 animate-pulse space-y-4">
        <div className="h-3 bg-slate-800 rounded w-1/3" />
        <div className="h-8 bg-slate-800 rounded w-2/3" />
        <div className="h-3 bg-slate-800 rounded" />
        <div className="h-3 bg-slate-800 rounded w-5/6" />
      </div>
    );
  }

  if (isError || !lesson) {
    return (
      <div className="max-w-3xl mx-auto px-8 py-12">
        <p className="text-red-400 mb-4">Lesson not found.</p>
        <Link to="/" className="text-indigo-400 hover:text-indigo-300">← Back to dashboard</Link>
      </div>
    );
  }

  const lessonProgress = progress?.lessons[lessonId];
  const isCompleted = lessonProgress?.status === 'completed';
  const quizAnswers = lessonProgress?.quizAnswers ?? {};

  return (
    <article className="px-8 py-12">
      {/* Breadcrumb */}
      <nav className="text-xs text-slate-500 mb-5 flex items-center gap-2 flex-wrap">
        <Link to="/" className="hover:text-slate-300 transition-colors">Dashboard</Link>
        <span>·</span>
        <span>{lesson.semesterId.replace(/-/g, ' ')}</span>
        <span>·</span>
        <span>{lesson.moduleId.replace(/-/g, ' ')}</span>
      </nav>

      {/* Tags */}
      <div className="flex items-center gap-2 flex-wrap mb-6">
        {lesson.estimatedMinutes && (
          <span className="text-xs text-slate-400 bg-slate-800 px-3 py-1 rounded-full">
            {lesson.estimatedMinutes} min
          </span>
        )}
        {lesson.tags.map(tag => (
          <span key={tag} className="text-xs text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded-full">
            {tag}
          </span>
        ))}
      </div>

      {/* Title */}
      <h1 className="text-3xl font-extrabold tracking-tight text-slate-100 mb-10">
        {lesson.title}
      </h1>

      {/* Reader */}
      <LessonReader
        lesson={lesson}
        quizAnswers={quizAnswers}
        onQuizAnswer={(quizIndex, selectedOption) =>
          saveQuizAnswer.mutate({ lessonId, quizIndex, selectedOption })
        }
      />

      {/* Footer nav */}
      <div className="mt-16 pt-8 border-t border-white/5 flex items-center justify-between">
        <span>
          {lesson.prev && (
            <Link
              to={`/lesson/${lesson.prev.semId}/${lesson.prev.modId}/${lesson.prev.lessonId}`}
              className="text-slate-400 hover:text-slate-200 transition-colors text-sm flex items-center gap-2"
            >
              <span>←</span> {lesson.prev.title}
            </Link>
          )}
        </span>

        {isCompleted ? (
          <span className="text-emerald-400 text-sm flex items-center gap-1.5 font-medium">
            ✓ Completed
          </span>
        ) : (
          <button
            onClick={() => markComplete.mutate(lessonId)}
            disabled={markComplete.isPending}
            className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-indigo-500 to-violet-500 text-white font-semibold text-sm hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {lesson.next ? 'Mark Complete & Continue →' : 'Mark Complete ✓'}
          </button>
        )}

        <span>
          {lesson.next && isCompleted && (
            <Link
              to={`/lesson/${lesson.next.semId}/${lesson.next.modId}/${lesson.next.lessonId}`}
              className="text-indigo-400 hover:text-indigo-300 transition-colors text-sm flex items-center gap-2 font-medium"
            >
              {lesson.next.title} <span>→</span>
            </Link>
          )}
        </span>
      </div>
    </article>
  );
}
```

- [ ] **Step 2: Verify lesson page end-to-end**

Navigate to http://localhost:5173/lesson/s1-math-foundations/m3-information-theory/l1-entropy
Expected:
- Title: "Shannon Entropy & Information Content"
- KaTeX formula renders (not raw `$$`)
- InlineQuiz shows 4 options
- Selecting an option reveals explanation
- "Mark Complete" button works
- Refresh → button shows "✓ Completed"
- No console errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/LessonPage.tsx
git commit -m "feat: wire lesson page with reader, quiz persistence, and complete button"
```

---

## Chunk 4: Dashboard

### Task 10: Dashboard with real progress data

**Files:**
- Create: `frontend/src/components/dashboard/ResumeCard.tsx`
- Create: `frontend/src/components/dashboard/StatsRow.tsx`
- Create: `frontend/src/components/dashboard/UpNextList.tsx`
- Modify: `frontend/src/pages/DashboardPage.tsx`

- [ ] **Step 1: Create `frontend/src/components/dashboard/ResumeCard.tsx`**

```typescript
import { Link } from 'react-router-dom';
import type { Curriculum, ProgressData } from '../../types.ts';

interface ResumeTarget {
  semId: string; modId: string; lessonId: string; title: string;
  moduleName: string; semesterName: string; completionPct: number;
}

function findResumeTarget(curriculum: Curriculum, progress: ProgressData): ResumeTarget | null {
  for (const sem of curriculum.semesters) {
    if (sem.status !== 'available') continue;
    for (const mod of sem.modules) {
      if (mod.status !== 'available') continue;
      for (const lesson of mod.lessons) {
        if (lesson.status !== 'available') continue;
        const lp = progress.lessons[lesson.id];
        if (!lp || lp.status !== 'completed') {
          const done = mod.lessons.filter(l => progress.lessons[l.id]?.status === 'completed').length;
          return {
            semId: sem.id, modId: mod.id, lessonId: lesson.id, title: lesson.title,
            moduleName: mod.title, semesterName: sem.title,
            completionPct: Math.round((done / mod.lessons.length) * 100),
          };
        }
      }
    }
  }
  return null;
}

export function ResumeCard({ curriculum, progress }: { curriculum: Curriculum; progress: ProgressData }) {
  const target = findResumeTarget(curriculum, progress);

  if (!target) {
    return (
      <div className="p-6 rounded-xl bg-slate-800/50 border border-white/8 text-center">
        <span className="text-emerald-400 font-bold">All available lessons complete!</span>
        <p className="text-slate-400 text-sm mt-1">More content coming soon.</p>
      </div>
    );
  }

  return (
    <div className="p-6 rounded-xl bg-gradient-to-br from-indigo-500/10 to-violet-500/10 border border-indigo-500/20">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <p className="text-xs text-slate-400 mb-1">{target.semesterName} · {target.moduleName}</p>
          <h2 className="text-lg font-bold text-slate-100 truncate">{target.title}</h2>
          <div className="mt-3 flex items-center gap-3">
            <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 rounded-full transition-all"
                style={{ width: `${target.completionPct}%` }}
              />
            </div>
            <span className="text-xs text-slate-400 flex-shrink-0">{target.completionPct}%</span>
          </div>
        </div>
        <Link
          to={`/lesson/${target.semId}/${target.modId}/${target.lessonId}`}
          className="flex-shrink-0 px-5 py-2.5 rounded-lg bg-gradient-to-r from-indigo-500 to-violet-500 text-white font-semibold text-sm hover:opacity-90 transition-opacity"
        >
          Resume
        </Link>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create `frontend/src/components/dashboard/StatsRow.tsx`**

```typescript
import type { ProgressData } from '../../types.ts';

export function StatsRow({ stats }: { stats: ProgressData['stats'] }) {
  const items = [
    { label: 'Day Streak', value: `${stats.streak} 🔥` },
    { label: 'Lessons Done', value: stats.totalCompleted },
    {
      label: 'Last Active',
      value: stats.lastActivityAt
        ? new Date(stats.lastActivityAt).toLocaleDateString()
        : '—',
    },
    { label: 'XP Earned', value: stats.totalCompleted * 70 },
  ];

  return (
    <div className="grid grid-cols-2 gap-4">
      {items.map(({ label, value }) => (
        <div key={label} className="p-4 rounded-xl bg-slate-800/50 border border-white/5">
          <p className="text-xs text-slate-500 mb-1">{label}</p>
          <p className="text-xl font-extrabold text-slate-100">{value}</p>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Create `frontend/src/components/dashboard/UpNextList.tsx`**

```typescript
import { Link } from 'react-router-dom';
import type { Curriculum, ProgressData } from '../../types.ts';

export function UpNextList({ curriculum, progress }: { curriculum: Curriculum; progress: ProgressData }) {
  const upNext: Array<{ semId: string; modId: string; lessonId: string; title: string }> = [];
  let skippedFirst = false;

  outer: for (const sem of curriculum.semesters) {
    if (sem.status !== 'available') continue;
    for (const mod of sem.modules) {
      if (mod.status !== 'available') continue;
      for (const lesson of mod.lessons) {
        if (lesson.status !== 'available') continue;
        const lp = progress.lessons[lesson.id];
        if (!lp || lp.status !== 'completed') {
          if (!skippedFirst) { skippedFirst = true; continue; }
          upNext.push({ semId: sem.id, modId: mod.id, lessonId: lesson.id, title: lesson.title });
          if (upNext.length >= 3) break outer;
        }
      }
    }
  }

  if (!upNext.length) return null;

  return (
    <div className="p-5 rounded-xl bg-slate-800/50 border border-white/5">
      <h3 className="text-sm font-semibold text-slate-400 mb-3">Up Next</h3>
      <ul className="space-y-2">
        {upNext.map(item => (
          <li key={item.lessonId}>
            <Link
              to={`/lesson/${item.semId}/${item.modId}/${item.lessonId}`}
              className="text-sm text-slate-300 hover:text-indigo-300 transition-colors flex items-center gap-2"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-500/60 flex-shrink-0" />
              {item.title}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 4: Replace `frontend/src/pages/DashboardPage.tsx`**

```typescript
import { useCurriculum } from '../hooks/useCurriculum.ts';
import { useProgress } from '../hooks/useProgress.ts';
import { ResumeCard } from '../components/dashboard/ResumeCard.tsx';
import { StatsRow } from '../components/dashboard/StatsRow.tsx';
import { UpNextList } from '../components/dashboard/UpNextList.tsx';

export default function DashboardPage() {
  const { data: curriculum, isLoading: loadingCurr } = useCurriculum();
  const { data: progress, isLoading: loadingProg } = useProgress();

  if (loadingCurr || loadingProg) {
    return (
      <div className="max-w-4xl mx-auto px-8 py-12 animate-pulse space-y-4">
        <div className="h-6 bg-slate-800 rounded w-1/4" />
        <div className="h-28 bg-slate-800 rounded" />
      </div>
    );
  }

  if (!curriculum || !progress) return null;

  return (
    <div className="max-w-4xl mx-auto px-8 py-12 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-extrabold tracking-tight text-slate-100">Welcome back</h1>
        {progress.stats.streak > 0 && (
          <span className="text-sm font-bold text-amber-400">{progress.stats.streak} 🔥</span>
        )}
      </div>

      <ResumeCard curriculum={curriculum} progress={progress} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <StatsRow stats={progress.stats} />
        <UpNextList curriculum={curriculum} progress={progress} />
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Verify dashboard end-to-end**

Open http://localhost:5173
Expected:
- Resume card shows first lesson with progress bar and Resume button
- Stats show streak, lessons done, XP
- Up Next shows next 2-3 lessons
- After completing a lesson: sidebar shows ✓, dashboard stats update, Resume card advances

- [ ] **Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat: add dashboard with resume card, stats, and up-next list"
```

---

## Chunk 5: Full Content

### Task 11: Full curriculum.json with all 8 semesters

**Files:**
- Modify: `content/curriculum.json`

- [ ] **Step 1: Replace `content/curriculum.json` with full 8-semester structure**

```json
{
  "semesters": [
    {
      "id": "s1-math-foundations",
      "title": "Semester 1: Mathematical Foundations",
      "description": "The mathematical language underpinning all modern ML.",
      "status": "available",
      "modules": [
        {
          "id": "m3-information-theory",
          "title": "Information Theory for ML",
          "description": "Entropy, KL divergence, mutual information, rate-distortion, and the information bottleneck.",
          "status": "available",
          "lessons": [
            { "id": "l1-entropy", "title": "Shannon Entropy & Information Content", "estimatedMinutes": 25, "status": "available" },
            { "id": "l2-kl-divergence", "title": "KL Divergence & f-Divergences", "estimatedMinutes": 30, "status": "available" },
            { "id": "l3-mutual-information", "title": "Mutual Information & Data Processing Inequality", "estimatedMinutes": 35, "status": "available" },
            { "id": "l4-rate-distortion", "title": "Rate-Distortion Theory", "estimatedMinutes": 30, "status": "available" },
            { "id": "l5-info-bottleneck", "title": "The Information Bottleneck Principle", "estimatedMinutes": 35, "status": "available" }
          ]
        },
        {
          "id": "m1-linear-algebra",
          "title": "Advanced Linear Algebra & Matrix Theory",
          "description": "SVD, random matrix theory, tensor algebra, matrix calculus.",
          "status": "coming-soon",
          "releaseNote": "Coming soon",
          "lessons": []
        },
        {
          "id": "m2-probability-theory",
          "title": "Measure-Theoretic Probability",
          "description": "Sigma-algebras, Lebesgue integration, martingales, stochastic processes.",
          "status": "coming-soon",
          "releaseNote": "Coming soon",
          "lessons": []
        },
        {
          "id": "m4-optimization-theory",
          "title": "Optimization Theory",
          "description": "Convex analysis, duality, proximal methods, non-convex landscapes.",
          "status": "coming-soon",
          "releaseNote": "Coming soon",
          "lessons": []
        },
        {
          "id": "m5-optimal-transport",
          "title": "Optimal Transport",
          "description": "Wasserstein distances, displacement interpolation, generative modeling applications.",
          "status": "coming-soon",
          "releaseNote": "Coming soon",
          "lessons": []
        },
        {
          "id": "m6-functional-analysis",
          "title": "Functional Analysis & Operator Theory",
          "description": "Hilbert spaces, RKHS, spectral theory — foundations of kernel methods and attention.",
          "status": "coming-soon",
          "releaseNote": "Coming soon",
          "lessons": []
        }
      ]
    },
    {
      "id": "s2-classical-ml",
      "title": "Semester 2: Advanced Classical ML",
      "description": "Probabilistic graphical models, kernel methods, Gaussian processes, generalization theory.",
      "status": "coming-soon",
      "modules": []
    },
    {
      "id": "s3-deep-learning",
      "title": "Semester 3: Deep Learning Theory & Practice",
      "description": "Optimization landscapes, expressivity, normalization, attention, scaling laws.",
      "status": "coming-soon",
      "modules": []
    },
    {
      "id": "s4-deep-rl",
      "title": "Semester 4: Deep Reinforcement Learning",
      "description": "Policy gradients, actor-critic, model-based RL, offline RL, RLHF.",
      "status": "coming-soon",
      "modules": []
    },
    {
      "id": "s5-generative-ai",
      "title": "Semester 5: Generative AI",
      "description": "VAEs, normalizing flows, GANs, diffusion models — theory and applications.",
      "status": "coming-soon",
      "modules": []
    },
    {
      "id": "s6-llms",
      "title": "Semester 6: Large Language Models",
      "description": "Transformer architecture, pretraining, instruction tuning, RLHF, inference, deployment.",
      "status": "coming-soon",
      "modules": []
    },
    {
      "id": "s7-gnns-causal-mlops",
      "title": "Semester 7: Graph Networks, Causality & MLOps",
      "description": "GNNs, causal inference for ML, production systems, monitoring.",
      "status": "coming-soon",
      "modules": []
    },
    {
      "id": "s8-frontier-topics",
      "title": "Semester 8: Frontier Topics",
      "description": "World models, embodied AI, AI safety, mechanistic interpretability, research frontiers.",
      "status": "coming-soon",
      "modules": []
    }
  ]
}
```

- [ ] **Step 2: Verify all 8 semesters appear in sidebar**

Open browser. Sidebar should show all 8 semesters. S1 expanded with 5 lesson links. S2–S8 as "coming soon".

- [ ] **Step 3: Commit**

```bash
git add content/curriculum.json
git commit -m "feat: expand curriculum to all 8 semesters"
```

---

### Task 12: Create module.json files + write 5 deep Information Theory lessons

**Files:**
- Create: `content/s1-math-foundations/m3-information-theory/module.json`
- Create: `content/s1-math-foundations/m1-linear-algebra/module.json`
- Create: `content/s1-math-foundations/m2-probability-theory/module.json`
- Create: `content/s1-math-foundations/m4-optimization-theory/module.json`
- Create: `content/s1-math-foundations/m5-optimal-transport/module.json`
- Create: `content/s1-math-foundations/m6-functional-analysis/module.json`
- Modify: `content/s1-math-foundations/m3-information-theory/l1-entropy.md`
- Create: `content/s1-math-foundations/m3-information-theory/l2-kl-divergence.md`
- Create: `content/s1-math-foundations/m3-information-theory/l3-mutual-information.md`
- Create: `content/s1-math-foundations/m3-information-theory/l4-rate-distortion.md`
- Create: `content/s1-math-foundations/m3-information-theory/l5-info-bottleneck.md`

- [ ] **Step 0: Create module.json for each S1 module**

These are metadata files per the repo layout spec. The backend does not read them at runtime — they serve as documentation and future extensibility.

`content/s1-math-foundations/m3-information-theory/module.json`:
```json
{
  "id": "m3-information-theory",
  "title": "Information Theory for ML",
  "status": "available"
}
```

Repeat for `m1-linear-algebra`, `m2-probability-theory`, `m4-optimization-theory`, `m5-optimal-transport`, `m6-functional-analysis` with their respective ids, titles, and `"status": "coming-soon"`.

**BEFORE writing lesson content:** Read `Research/advanced_outline.md` and `Research/Gemini.md` for source material.

Each lesson: 1500–2500 words, YAML frontmatter, graduate-level prose, `$...$` and `$$...$$` KaTeX math, annotated Python code blocks, 2–3 `:::quiz` knowledge checks, key insight blockquotes.

- [ ] **Step 1: Read research files for source material**

Run:
```bash
cat Research/advanced_outline.md
cat Research/Gemini.md
```

Use these as authoritative sources for mathematical content and ML connections.

- [ ] **Step 2: Write l1-entropy.md**

Key topics: self-information (surprisal), Shannon entropy derivation, joint/conditional entropy, chain rule, cross-entropy as neural network loss, bits vs nats, source coding theorem interpretation.

Frontmatter:
```yaml
---
title: "Shannon Entropy & Information Content"
estimatedMinutes: 25
tags: ["information-theory", "entropy", "cross-entropy", "shannon"]
prerequisites: []
---
```

- [ ] **Step 3: Write l2-kl-divergence.md**

Key topics: KL divergence definition, forward vs reverse KL asymmetry (mode-seeking vs mode-covering), total variation distance, Hellinger distance, chi-squared divergence, Donsker-Varadhan variational representation, why KL matters for variational inference and VAEs.

Frontmatter:
```yaml
---
title: "KL Divergence & f-Divergences"
estimatedMinutes: 30
tags: ["kl-divergence", "variational-inference", "f-divergences", "VAE"]
prerequisites: ["l1-entropy"]
---
```

- [ ] **Step 4: Write l3-mutual-information.md**

Key topics: MI definition, relationships to entropy (I(X;Y) = H(X) - H(X|Y)), chain rules, data processing inequality (formal statement + proof sketch), channel capacity, MI in representation learning (InfoNCE, contrastive objectives), MINE estimator.

Frontmatter:
```yaml
---
title: "Mutual Information & Data Processing Inequality"
estimatedMinutes: 35
tags: ["mutual-information", "DPI", "channel-capacity", "representation-learning", "MINE"]
prerequisites: ["l1-entropy", "l2-kl-divergence"]
---
```

- [ ] **Step 5: Write l4-rate-distortion.md**

Key topics: rate-distortion function R(D) definition, R(D) for Gaussian sources (closed form), the trade-off curve and its interpretation, connection between the VAE latent bottleneck and R-D, model compression as R-D, neural image compression.

Frontmatter:
```yaml
---
title: "Rate-Distortion Theory"
estimatedMinutes: 30
tags: ["rate-distortion", "VAE", "model-compression", "neural-compression"]
prerequisites: ["l1-entropy", "l2-kl-divergence", "l3-mutual-information"]
---
```

- [ ] **Step 6: Write l5-info-bottleneck.md**

Key topics: Tishby's IB formulation (minimize I(X;T) while maximizing I(T;Y)), the IB Lagrangian, IB in deep networks (Tishby-Schwartz-Bialek 2017 paper), the "fitting and compression" phases debate, practical implications for representation quality, connections to contrastive SSL.

Frontmatter:
```yaml
---
title: "The Information Bottleneck Principle"
estimatedMinutes: 35
tags: ["information-bottleneck", "representation-learning", "deep-learning-theory", "SSL"]
prerequisites: ["l1-entropy", "l2-kl-divergence", "l3-mutual-information"]
---
```

- [ ] **Step 7: Verify all 5 lessons render correctly**

Navigate to each lesson. Confirm:
- Math renders via KaTeX (no raw `$$` showing)
- Code blocks are syntax-highlighted with copy button
- 2–3 quizzes per lesson work (select → explanation reveals)
- Prev/Next navigation works across all 5 lessons
- No console errors

- [ ] **Step 8: Commit**

```bash
git add content/s1-math-foundations/m3-information-theory/
git commit -m "feat: write 5 deep Information Theory lessons (S1/M1.3)"
```

---

## Chunk 6: Verification

### Task 13: Final verification against all 8 success criteria

- [ ] **Criterion 1: Sidebar renders all 8 semesters, S1/M1.3 lessons are linked and navigable**

Open http://localhost:5173. Verify 8 semester sections. Click all 5 Information Theory lessons.
Expected: Navigates correctly each time.

- [ ] **Criterion 2: "Coming soon" items show correct stub UI**

Inspect S2–S8 sidebar items. Confirm they have "coming soon" badges and are NOT links.
Expected: cursor:default on hover, no navigation.

- [ ] **Criterion 3: Information Theory module renders fully (prose + KaTeX + code + quizzes)**

Open each of the 5 lessons. Confirm math renders (check l2 for KL formula), code block highlights Python, quizzes interactive.
Expected: All content renders correctly, no raw `$$` visible.

- [ ] **Criterion 4: Quiz selection works and persists across refresh**

Pick a quiz answer on any lesson. Refresh (Ctrl+R).
Expected: Answer still selected, explanation still visible.

- [ ] **Criterion 5: Progress persists after reload**

Click "Mark Complete" on l1-entropy. Reload.
Expected: Sidebar shows ✓ next to l1-entropy. Dashboard stats update.

- [ ] **Criterion 6: Dashboard shows correct resume lesson, streak, stats**

From fresh session, complete 2 lessons. Go to dashboard (http://localhost:5173).
Expected: Resume card points to lesson 3. Stats show correct count.

- [ ] **Criterion 7: `pnpm dev` from root starts both services**

Kill all processes. From repo root:
```bash
pnpm dev
```
Expected: Backend logs "Backend running on http://localhost:3001" AND frontend Vite starts on port 5173. No manual steps needed.

- [ ] **Criterion 8: No console errors on any page**

Open DevTools (F12) → Console. Navigate to: dashboard, l1-entropy lesson, l3-mutual-information lesson, SemesterPage (http://localhost:5173/semester/s2-classical-ml).
Expected: Zero red errors in console.

- [ ] **Final commit**

```bash
git add .
git commit -m "feat: complete AI Mastery Hub MVP — all 8 success criteria verified"
```

---

## File Map Summary

### Backend (`backend/src/`)
| File | Responsibility |
|---|---|
| `app.ts` | Express setup, CORS, route mounting |
| `lib/content.ts` | Load `curriculum.json`, read raw lesson markdown |
| `lib/quiz.ts` | Split markdown at `:::quiz` blocks via remark AST (unist-util-visit), parse YAML |
| `lib/db.ts` | SQLite schema, progress CRUD, streak calculation |
| `routes/curriculum.ts` | `GET /api/curriculum` |
| `routes/lessons.ts` | `GET /api/lessons/:sem/:mod/:lesson` — parses frontmatter + content |
| `routes/progress.ts` | `GET /api/progress`, `POST /api/progress/:lessonId` |
| `tests/content.test.ts` | Tests for loadCurriculum |
| `tests/quiz.test.ts` | Tests for splitContent |
| `tests/progress.test.ts` | Tests for db CRUD (in-memory SQLite) |

### Frontend (`frontend/src/`)
| File | Responsibility |
|---|---|
| `types.ts` | Shared TypeScript interfaces (mirrors backend shapes) |
| `lib/api.ts` | Typed fetch wrapper — injects X-Session-Id header |
| `lib/session.ts` | localStorage UUID + first-visit flag |
| `hooks/useCurriculum.ts` | TanStack Query → GET /api/curriculum |
| `hooks/useLesson.ts` | TanStack Query → GET /api/lessons/... |
| `hooks/useProgress.ts` | TanStack Query + mutations → GET/POST /api/progress |
| `components/layout/AppShell.tsx` | Fixed layout shell with outlet |
| `components/layout/Header.tsx` | 48px top bar with logo |
| `components/layout/Sidebar.tsx` | 280px curriculum tree — `SidebarItem` / `LessonItem` logic co-located here, reads completedLessons |
| `components/reader/LessonReader.tsx` | Renders content[] blocks (markdown + quiz) |
| `components/reader/CodeBlock.tsx` | Syntax highlighting with copy button |
| `components/reader/InlineQuiz.tsx` | MCQ with answer reveal + explanation |
| `components/dashboard/ResumeCard.tsx` | Resume button with progress bar |
| `components/dashboard/StatsRow.tsx` | Streak, lessons done, XP, last active |
| `components/dashboard/UpNextList.tsx` | Next 3 available incomplete lessons |
| `components/ErrorBoundary.tsx` | React error boundary |
| `components/Toast.tsx` | First-visit notice |
| `pages/DashboardPage.tsx` | Dashboard — resume + stats + up next |
| `pages/SemesterPage.tsx` | Coming soon stub |
| `pages/LessonPage.tsx` | Full lesson view — reader + nav + complete |
| `App.tsx` | Route definitions |

### Content (`content/`)
| Path | Responsibility |
|---|---|
| `curriculum.json` | Authoritative curriculum index — 8 semesters |
| `s1-math-foundations/m3-information-theory/l*.md` | 5 deep lessons (Phase 5) |
