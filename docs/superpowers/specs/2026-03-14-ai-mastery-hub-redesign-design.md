# AI Mastery Hub — Platform Redesign Spec
**Date:** 2026-03-14
**Status:** Approved for implementation

---

## 1. Vision

A world-class graduate-level AI/ML learning platform that replaces the need to navigate years of scattered papers, courses, and textbooks. The platform's north star: **content quality first, everything else in service of reading deeply**.

Target audience: engineers, researchers, and graduate students who are past the basics and want rigorous, research-grade mastery across the full breadth of modern AI.

---

## 2. Design Decisions (Approved)

| Decision | Choice | Rationale |
|---|---|---|
| Visual personality | Bold & Modern | Dark base, vivid gradient accents, high energy. Feels like a premium product. |
| Navigation | Persistent sidebar | Always-visible curriculum tree. Users always know where they are. |
| Lesson layout | Deep Reader | Wide content column, theory → math → code flows top-to-bottom. |
| Dashboard | Combined | Resume button prominent, stats left, upcoming lessons right. |
| Interactivity | Inline knowledge checks | Embedded multiple-choice, no browser code execution in MVP. |

---

## 3. Architecture

### Philosophy
Simplicity over sophistication. Content is the product. The architecture serves content, not the other way around.

### Stack

```
Frontend:  React 18 + TypeScript + Vite        (port 5173 in dev)
Styling:   Tailwind CSS v3
Routing:   React Router v6
State:     TanStack Query (server) + useState (local UI state)
Math:      react-katex (client-side KaTeX rendering — see §3.4)
Code:      react-syntax-highlighter (Prism)
Markdown:  react-markdown + remark-math + rehype-katex + remark-directive

Backend:   Express.js + TypeScript              (port 3001 in dev)
Content:   Markdown files on disk, read at request time (no DB for content)
Progress:  better-sqlite3 (file: backend/data/progress.db)
```

### Repository Layout

New code lives at the root in `frontend/` and `backend/`. The existing `artifacts/` directory is **renamed to `artifacts-archived/`** (one git mv, no deletion) so history is preserved but it's clearly out of service.

```
AI-Mastery-Hub/
├── frontend/                        # React app (NEW)
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/              # AppShell, Sidebar, SidebarItem, Header
│   │   │   ├── reader/              # LessonReader, MathBlock, CodeBlock, InlineQuiz
│   │   │   └── dashboard/           # ResumeCard, StatsRow, UpNextList
│   │   ├── pages/
│   │   │   ├── DashboardPage.tsx
│   │   │   ├── SemesterPage.tsx     # "Coming Soon" view for locked semesters
│   │   │   └── LessonPage.tsx
│   │   ├── hooks/
│   │   │   ├── useCurriculum.ts     # TanStack Query → GET /api/curriculum
│   │   │   ├── useLesson.ts         # TanStack Query → GET /api/lessons/...
│   │   │   └── useProgress.ts       # TanStack Query → GET/POST /api/progress
│   │   ├── lib/
│   │   │   ├── api.ts               # Typed fetch wrapper (base: /api)
│   │   │   └── session.ts           # localStorage UUID management
│   │   └── main.tsx
│   ├── index.html
│   └── vite.config.ts               # proxy /api → localhost:3001
│
├── backend/                         # Express API (NEW)
│   ├── src/
│   │   ├── routes/
│   │   │   ├── curriculum.ts        # GET /api/curriculum
│   │   │   ├── lessons.ts           # GET /api/lessons/:sem/:mod/:lesson
│   │   │   └── progress.ts          # GET /api/progress, POST /api/progress/:lessonId
│   │   ├── lib/
│   │   │   ├── content.ts           # Reads & parses markdown, indexes curriculum.json
│   │   │   ├── quiz.ts              # Parses :::quiz directive blocks from markdown AST
│   │   │   └── db.ts                # SQLite schema + queries
│   │   └── app.ts                   # Express setup, CORS, routes
│   ├── data/                        # Gitignored — progress.db lives here at runtime
│   └── tsconfig.json
│
├── content/                         # Curriculum (NEW — markdown files)
│   ├── curriculum.json              # Authoritative index (see §3.3)
│   ├── s1-math-foundations/
│   │   ├── m1-linear-algebra/
│   │   │   ├── module.json
│   │   │   ├── l1-svd-low-rank.md
│   │   │   ├── l2-random-matrix-theory.md
│   │   │   ├── l3-tensor-algebra.md
│   │   │   ├── l4-matrix-calculus.md
│   │   │   └── l5-quiz.md
│   │   ├── m2-probability-theory/ (stub)
│   │   ├── m3-information-theory/   # ← DEEP CONTENT (5 lessons, see §5)
│   │   ├── m4-optimization-theory/ (stub)
│   │   ├── m5-optimal-transport/ (stub)
│   │   └── m6-functional-analysis/ (stub)
│   ├── s2-classical-ml/ (stub)
│   ├── s3-deep-learning/ (stub)
│   ├── s4-deep-rl/ (stub)
│   ├── s5-generative-ai/ (stub)
│   ├── s6-llms/ (stub)
│   ├── s7-gnns-causal-mlops/ (stub)
│   └── s8-frontier-topics/ (stub)
│
├── artifacts-archived/              # Renamed from artifacts/ — preserved, not used
├── docs/
│   └── superpowers/specs/
│       └── 2026-03-14-ai-mastery-hub-redesign-design.md  ← this file
├── package.json                     # Root — pnpm workspace scripts only
└── pnpm-workspace.yaml              # workspaces: [frontend, backend]
```

### Content Addressing (IDs)

All IDs are **kebab-case slugs** matching their directory names. This keeps filesystem, API URLs, and React Router paths in sync.

| Level | ID example | Directory name |
|---|---|---|
| Semester | `s1-math-foundations` | `content/s1-math-foundations/` |
| Module | `m3-information-theory` | `.../m3-information-theory/` |
| Lesson | `l1-entropy` | `.../l1-entropy.md` |

React Router path: `/lesson/s1-math-foundations/m3-information-theory/l1-entropy`
API path: `GET /api/lessons/s1-math-foundations/m3-information-theory/l1-entropy`

### curriculum.json Structure

`status` exists at both the **module level** and the **lesson level**. This is what determines whether sidebar items are linked.

- **Module `status: "coming-soon"`** → the module heading in the sidebar is not linked; no lesson items are shown
- **Module `status: "available"`** → lessons are listed; each lesson's own `status` controls linking
- **Lesson `status: "available"`** → rendered as a clickable link in the sidebar
- **Lesson `status: "coming-soon"`** → rendered as muted text, not linked

```json
{
  "semesters": [
    {
      "id": "s1-math-foundations",
      "title": "Semester 1: Mathematical Foundations",
      "description": "The mathematical language underpinning all modern ML...",
      "status": "available",
      "modules": [
        {
          "id": "m3-information-theory",
          "title": "Information Theory for ML",
          "description": "Entropy, KL divergence, mutual information...",
          "status": "available",
          "lessons": [
            { "id": "l1-entropy", "title": "Shannon Entropy & Information Content", "estimatedMinutes": 25, "status": "available" },
            { "id": "l2-kl-divergence", "title": "KL Divergence & f-Divergences", "estimatedMinutes": 30, "status": "available" }
          ]
        },
        {
          "id": "m1-linear-algebra",
          "title": "Advanced Linear Algebra & Matrix Theory",
          "status": "coming-soon",
          "releaseNote": "Coming in Phase 2",
          "lessons": []
        }
      ]
    },
    {
      "id": "s2-classical-ml",
      "title": "Semester 2: Advanced Classical ML",
      "status": "coming-soon",
      "modules": []
    }
  ]
}
```

### Math Rendering

KaTeX renders **client-side** in the React frontend. The backend serves raw markdown text. The frontend pipeline is:

```
raw markdown string
  → react-markdown
    + remark-math       (parses $...$ and $$...$$)
    + rehype-katex      (renders to HTML via KaTeX)
  → rendered React tree
```

This keeps the backend simple (no HTML generation) and gives full KaTeX feature support in the browser.

### Quiz Syntax & Parser

Lesson markdown uses `remark-directive` container directives. The body of a `:::quiz` block is **YAML** (parsed by `js-yaml` inside `quiz.ts`):

```markdown
:::quiz
question: "Which distribution maximizes entropy for 3 outcomes?"
options:
  - "[0.5, 0.3, 0.2]"
  - "[1/3, 1/3, 1/3]"
  - "[0.9, 0.05, 0.05]"
  - "[0.6, 0.4, 0.0]"
correct: 1
explanation: "The uniform distribution maximizes entropy. H is maximized when all outcomes are equally likely."
:::
```

Field rules:
- `question`: string (required)
- `options`: array of strings, 2–6 items (required)
- `correct`: zero-based integer index into `options` (required)
- `explanation`: string shown after answer is selected (required)

`quiz.ts` walks the remark AST for `containerDirective` nodes named `"quiz"`, parses their text content as YAML, and validates all four fields are present. Invalid quiz blocks log a warning and are omitted from output rather than crashing.

The backend splits each lesson into a `content` array at `:::quiz` boundaries. Each quiz block becomes a `{ type: "quiz", ... }` element; surrounding markdown becomes `{ type: "markdown", raw: string }` elements. Multiple quiz blocks per lesson are fully supported.

The frontend `InlineQuiz` component receives parsed quiz data and handles selection/reveal logic entirely client-side (no round-trip on answer).

### Content Path Resolution

`content/` lives at the **repo root**, outside `backend/`. The backend resolves it via an environment variable `CONTENT_DIR`, with a default that works in both dev and production:

```ts
// backend/src/lib/content.ts
const CONTENT_DIR = process.env.CONTENT_DIR
  ?? path.resolve(__dirname, '../../../content');  // backend/src/lib → repo root
```

In dev, this resolves correctly from the compiled output. For production, set `CONTENT_DIR` explicitly. This is documented in `backend/.env.example`.

### API Contracts

All API responses use a consistent envelope:

```json
// Success
{ "ok": true, "data": { /* payload */ } }

// Error
{ "ok": false, "error": { "code": "NOT_FOUND", "message": "Lesson not found" } }
```

Error codes: `NOT_FOUND` (404), `BAD_REQUEST` (400), `INTERNAL_ERROR` (500). The frontend checks `ok` before accessing `data`.

---

**`GET /api/curriculum`**
```json
{
  "ok": true,
  "data": {
    "semesters": [ /* full semester tree as defined in §3.3 */ ]
  }
}
```

**`GET /api/lessons/:semId/:modId/:lessonId`**

`tags` and `prerequisites` come from the lesson markdown **YAML frontmatter**:
```markdown
---
title: "KL Divergence & f-Divergences"
estimatedMinutes: 30
tags: ["variational-inference", "VAE", "divergences"]
prerequisites: ["l1-entropy"]
---
```

Response:
```json
{
  "ok": true,
  "data": {
    "id": "l2-kl-divergence",
    "title": "KL Divergence & f-Divergences",
    "semesterId": "s1-math-foundations",
    "moduleId": "m3-information-theory",
    "estimatedMinutes": 30,
    "tags": ["variational-inference", "VAE", "divergences"],
    "prerequisites": ["l1-entropy"],
    "content": [
      { "type": "markdown", "raw": "## Overview\n\nThe KL divergence..." },
      { "type": "quiz", "question": "Which of these...", "options": ["A","B","C","D"], "correct": 1, "explanation": "..." },
      { "type": "markdown", "raw": "## Why It Matters\n\n..." }
    ],
    "prev": { "semId": "s1-math-foundations", "modId": "m3-information-theory", "lessonId": "l1-entropy", "title": "Shannon Entropy & Information Content" },
    "next": { "semId": "s1-math-foundations", "modId": "m3-information-theory", "lessonId": "l3-mutual-information", "title": "Mutual Information & Data Processing Inequality" }
  }
}
```

`prev` and `next` are `null` for the first/last lesson respectively. Shape is always `{ semId, modId, lessonId, title }` or `null`.

`"markdown"` blocks contain raw markdown with `$...$` and `$$...$$` which KaTeX renders client-side. The split happens at `:::quiz` boundaries.

**`GET /api/progress`**
Requires `X-Session-Id` header.
```json
{
  "ok": true,
  "data": {
    "lessons": {
      "l1-entropy": { "status": "completed", "completedAt": "2026-03-10T14:22:00Z", "quizAnswers": {} },
      "l2-kl-divergence": { "status": "in_progress", "completedAt": null, "quizAnswers": { "0": 1 } }
    },
    "stats": {
      "streak": 7,
      "totalCompleted": 12,
      "lastActivityAt": "2026-03-14T09:00:00Z"
    }
  }
}
```

**`POST /api/progress/:lessonId`**
Requires `X-Session-Id` header. Exactly one of `status` or `quizAnswer` must be present.
```json
// Mark lesson complete
{ "status": "completed" }

// Save a quiz answer (does not change lesson status)
{ "quizAnswer": { "quizIndex": 0, "selectedOption": 1 } }

// Response (both cases)
{ "ok": true, "data": null }
```

---

## 4. UI Design

### Color Palette

```
Background:      #0f172a  (slate-900)
Surface:         #1e293b  (slate-800)
Surface deep:    #0a0f1e  (custom)
Border subtle:   rgba(255,255,255,0.06)
Accent primary:  #6366f1  (indigo-500)
Accent second:   #8b5cf6  (violet-500)
Gradient:        linear-gradient(135deg, #6366f1, #8b5cf6)
Text primary:    #f1f5f9  (slate-100)
Text secondary:  #94a3b8  (slate-400)
Text muted:      #475569  (slate-600)
Success:         #10b981  (emerald-500)
Warning:         #f59e0b  (amber-500)
Math accent:     #a5b4fc  (indigo-300)
```

### Typography
- Headings: `font-extrabold tracking-tight`
- Body prose: `leading-relaxed` (1.75), max ~70 characters wide
- Code: `font-mono` (system mono stack)
- Math: KaTeX default (publication-quality)

### App Layout

```
┌─────────────────────────────────────────────────────┐
│  Header (48px): Logo + breadcrumb + global progress │
├──────────────┬──────────────────────────────────────┤
│              │                                      │
│  Sidebar     │  Main area                           │
│  (280px      │  (full remaining width)              │
│  fixed)      │                                      │
│              │  Dashboard | SemesterPage | Lesson   │
│              │                                      │
└──────────────┴──────────────────────────────────────┘
```

Sidebar is fixed-position, scrollable independently. Main area scrolls normally.

### Sidebar

```
[AI Mastery Hub logo / wordmark]

▼ Semester 1: Math Foundations    ███░░ 38%
  ▼ M1.3 Information Theory       ← expanded (active)
      ✓ Shannon Entropy
      ● KL Divergence              ← active lesson
        Mutual Information
        Rate-Distortion
        Information Bottleneck
  ► M1.1 Linear Algebra           [coming soon]
  ► M1.2 Probability Theory       [coming soon]
  ► M1.4 Optimization Theory      [coming soon]
  ► M1.5 Optimal Transport        [coming soon]
  ► M1.6 Functional Analysis      [coming soon]

► Semester 2: Classical ML        [coming soon]
► Semester 3: Deep Learning       [coming soon]
...
► Semester 8: Frontier Topics     [coming soon]
```

Sidebar items for "coming soon" modules/semesters are rendered but **not linked** (cursor-default, muted color, "coming soon" badge). Clicking them does nothing, keeping routing clean.

### Lesson Reader

```
[Breadcrumb: S1 · Information Theory · KL Divergence]
[Pill: Lesson 2 of 5 · 30 min]

# KL Divergence & f-Divergences

───────────────────────────────────────────
[Section: overview prose]

[Section: theory prose + math block]

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

[Key insight callout — indigo left-border]

[Code block — syntax highlighted, copy button]

[InlineQuiz component]

[More prose sections...]
───────────────────────────────────────────

[← Previous: Shannon Entropy]   [Mark Complete & Continue →]
```

### Dashboard

```
Welcome back                                    7🔥 streak

┌─────────────────────────────────────────────────────┐
│  ▶  KL Divergence & f-Divergences         [Resume] │
│     Semester 1 · Information Theory · 58% done     │
│     ████████████░░░░░░░░                           │
└─────────────────────────────────────────────────────┘

┌───────────────────────┐  ┌───────────────────────────┐
│  Streak:   7 days  🔥 │  │  Up Next                  │
│  Lessons done:    12  │  │  • Mutual Information      │
│  Sem. progress: 1/8   │  │  • Rate-Distortion Theory  │
│  XP:            840   │  │  • Information Bottleneck  │
└───────────────────────┘  └───────────────────────────┘
```

---

## 5. Content Plan

### Phase 4 (all stubs): Every semester and module has at minimum:
- Title, short description, topic bullets
- "Coming Soon" status with an encouraging message

### Phase 5 (deep content): S1 / M1.3 — Information Theory for ML

Five fully written lessons based on `Research/advanced_outline.md` and `Research/Gemini.md`:

| ID | Title | Key topics |
|---|---|---|
| `l1-entropy` | Shannon Entropy & Information Content | Self-information, H(X) derivation, cross-entropy, bits vs nats, coding interpretation |
| `l2-kl-divergence` | KL Divergence & f-Divergences | Forward/reverse KL, total variation, Hellinger, chi-squared, variational representations, intuition |
| `l3-mutual-information` | Mutual Information & Data Processing Inequality | MI definition, DPI, channel capacity, MI in representation learning, MINE estimator |
| `l4-rate-distortion` | Rate-Distortion Theory | R-D function, trade-off curve, connection to VAE latent bottleneck, model compression |
| `l5-info-bottleneck` | The Information Bottleneck Principle | Tishby's IB, IB Lagrangian, IB in deep learning debate, mutual information minimization |

Each lesson targets ~1500–2500 words of graduate-level prose, properly typeset math, annotated Python snippets, 2–3 inline knowledge checks, key insight callouts, and explicit connections to modern ML architectures.

---

## 6. Progress Tracking

### Session Management

- On first page load, `session.ts` checks localStorage for `amh_session_id`
- If absent, generates a UUID v4, stores it, and sets `amh_first_visit = "true"` in localStorage
- Every API request sends the UUID as `X-Session-Id` header
- **"First visit" definition**: `amh_first_visit` key is absent from localStorage (i.e., a brand-new browser or cleared storage). The toast is shown once, then `amh_first_visit` is written so it never shows again on the same browser.
- **Edge cases (policy)**:
  - Cleared localStorage → new UUID + new session, prior progress unrecoverable. Toast re-appears informing the user. Accepted limitation for MVP.
  - Multiple tabs simultaneously → last write wins per `(session_id, lesson_id)`. SQLite serializes writes. Accepted for single-user MVP.
  - SQLite row accumulation → no cleanup. File size is negligible at MVP scale (each row ~200 bytes).

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS progress (
  session_id   TEXT NOT NULL,
  lesson_id    TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'not_started',  -- not_started | in_progress | completed
  quiz_answers TEXT,          -- JSON: { "0": 1, "1": 3 } — quizIndex → selectedOption
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
```

Streak is calculated server-side: if `last_streak_at` date differs from today by exactly 1 day, increment; if same day, no change; if > 1 day gap, reset to 1.

---

## 7. What's Explicitly NOT in MVP

| Feature | Decision |
|---|---|
| User auth / accounts | Deferred — session-based is sufficient |
| In-browser Python execution | Deferred — link to Colab instead |
| Gamification / badges | Deferred — content drives graduate-level retention |
| Search | Deferred — browser find works; add in Phase 2 |
| Comments / discussion | Deferred — needs auth + moderation |
| Mobile-optimized layout | Deferred — sidebar pattern works at tablet+; phone is Phase 2 |
| Multi-user / leaderboards | Out of scope |

---

## 8. Build Approach & Cleanup

**This is a greenfield build.** All code in this spec is written from scratch in `frontend/` and `backend/`. No code from `artifacts/`, `lib/`, or any other existing directory is carried forward. Field names, API shapes, error formats, and interfaces defined here supersede the old codebase entirely — do not reference old code for implementation guidance.

**Delete before starting Phase 1:**

| Path | Reason |
|---|---|
| `artifacts/` | Old frontend, backend, mockup-sandbox — all replaced |
| `lib/` | Over-engineered shared libs (api-client-react, api-zod, api-spec, db) — not used |
| `scripts/` | Replit deployment scripts — not relevant |
| `.agents/` | Empty Replit agents folder |
| `.local/` | Replit local agent state |
| `AUDIT_SUMMARY.md` | Previous agent doc — replaced by this spec |
| `DEVELOPMENT_PROGRESS.md` | Previous agent doc — replaced by implementation plan |
| `SPEC_QUICK_REFERENCE.md` | Previous agent doc |
| `README.md` | Replit-era readme — rewrite for new platform |
| `pnpm-workspace.yaml` | References `artifacts/*` and `lib/*` — rewrite |
| `tsconfig.base.json` | Old shared tsconfig — replaced per-package |
| `node_modules/` | Regenerated after cleanup |
| `pnpm-lock.yaml` | Regenerated after cleanup |

**Move before deleting:**
- `advanced_outline.md` (root) → `Research/advanced_outline.md` (already exists there; delete root copy)

**Preserve:**
- `Research/` — curriculum research, used as content source
- `docs/` — this spec
- `.git/`, `.gitignore`, `.vscode/`, `.claude/`

---

## 9. Dev Setup

```bash
# Root
pnpm install          # installs all workspaces

# Two terminals:
pnpm --filter frontend dev     # http://localhost:5173
pnpm --filter backend dev      # http://localhost:3001

# Or from root with concurrently:
pnpm dev              # runs both via root package.json "dev" script
```

**Vite proxy config** lives in `frontend/vite.config.ts`:
```ts
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:3001', changeOrigin: true }
    }
  }
})
```

The frontend always calls relative `/api/...` — no hardcoded backend URL anywhere in frontend code. In production, a reverse proxy (nginx or similar) routes `/api` to the Express process.

---

## 10. Success Criteria (MVP Complete When)

1. Sidebar renders all 8 semesters; S1/M1.3 lessons are linked and navigable
2. "Coming soon" modules/semesters show correct stub UI (not linked, descriptive message)
3. Information Theory module (5 lessons) renders fully: prose, KaTeX math, syntax-highlighted code, inline quizzes
4. Quiz selection works: choose an option, explanation is revealed, answer persists across refresh
5. Progress persists: complete a lesson, reload — sidebar shows it complete, dashboard reflects it
6. Dashboard shows correct resume lesson, streak, completion stats
7. `pnpm dev` from root starts both services with no manual steps
8. No console errors on lesson, dashboard, or semester pages

---

## 11. Implementation Phases

| Phase | Scope | Done when |
|---|---|---|
| 1 | Scaffold `frontend/` + `backend/`, wire up sidebar + routing, stub pages | App shell loads, sidebar renders from `curriculum.json`, routing works |
| 2 | Lesson reader: markdown → react-markdown pipeline, KaTeX, syntax highlighting, InlineQuiz component | Can load and read a lesson with math, code, and a working quiz |
| 3 | Progress API + SQLite, dashboard with real data | Dashboard shows real resume state; lesson complete button works and persists |
| 4 | Populate `content/` with all 8 semester stubs | Full curriculum arc visible in sidebar |
| 5 | Write S1/M1.3 Information Theory — 5 deep lessons | First module is complete and excellent |
| 6 | Visual polish: animations, sidebar transitions, responsive tweaks, empty states, error boundaries | Passes all 8 success criteria; ready to share |
