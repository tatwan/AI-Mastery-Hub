# Handoff Notes — AI Mastery Hub

**Last updated:** 2026-03-14
**Status:** Spec approved, project cleaned up, ready to begin Phase 1 implementation

---

## Where We Are

The brainstorming and design phase is **complete**. The spec is written, reviewed, and committed. The project has been cleaned of all previous-agent artifacts.

The repo currently contains:
- `docs/superpowers/specs/2026-03-14-ai-mastery-hub-redesign-design.md` — the authoritative spec
- `Research/` — 7 curriculum research files used as content source
- `package.json` + `pnpm-workspace.yaml` — clean root workspace config (only `frontend` + `backend`)
- `.gitignore`, `.vscode/`, `.claude/`, `README.md`

**Nothing else.** `frontend/`, `backend/`, and `content/` do not exist yet — they are built in Phase 1.

---

## Next Step: Invoke `writing-plans` skill

The brainstorming skill's terminal state is invoking the **`writing-plans`** skill to generate a detailed implementation plan from the spec. Do this before touching any code.

```
Skill: writing-plans
Input: docs/superpowers/specs/2026-03-14-ai-mastery-hub-redesign-design.md
```

---

## Implementation Phases (from spec §11)

| Phase | Scope | Done when |
|---|---|---|
| **1** | Scaffold `frontend/` + `backend/`, wire sidebar + routing, stub pages | App shell loads, sidebar renders from `curriculum.json`, routing works |
| **2** | Lesson reader: markdown pipeline, KaTeX, syntax highlighting, InlineQuiz | Can load and read a lesson with math, code, and a working quiz |
| **3** | Progress API + SQLite, dashboard with real data | Dashboard shows real resume state; complete button persists |
| **4** | Populate `content/` — all 8 semester stubs | Full curriculum arc visible in sidebar |
| **5** | Write S1/M1.3 Information Theory — 5 deep lessons | First module complete and excellent |
| **6** | Visual polish, animations, error boundaries | Passes all 8 success criteria; ready to share |

---

## Key Design Decisions (already locked — do not re-open)

| Decision | Choice |
|---|---|
| Visual personality | Bold & Modern — dark slate-900, indigo/violet gradient accents |
| Navigation | Persistent 280px sidebar |
| Lesson layout | Deep Reader (wide content column, theory → math → code) |
| Dashboard | Combined (resume button prominent, stats + upcoming) |
| Interactivity | Inline knowledge checks (no in-browser execution in MVP) |
| Content storage | Markdown files on disk, not a database |
| Content IDs | Kebab-case slugs matching directory names |
| Session auth | Anonymous — localStorage UUID, `X-Session-Id` header |
| API envelope | `{ ok, data }` / `{ ok: false, error: { code, message } }` |
| Deep content MVP | S1/M1.3 Information Theory — 5 lessons |

---

## Critical Implementation Details

- `content/` lives at **repo root**, not inside `backend/`
- Backend resolves content via: `CONTENT_DIR ?? path.resolve(__dirname, '../../../content')`
- Quiz syntax: `:::quiz` blocks with YAML body (`question`, `options[]`, `correct` int, `explanation`)
- Prev/next lesson shape: `{ semId, modId, lessonId, title } | null`
- Lesson `status` lives at **both** module AND lesson level in `curriculum.json`
- Frontend always calls relative `/api/...` — no hardcoded backend URL

---

## Success Criteria (MVP complete when all 8 pass)

1. Sidebar renders all 8 semesters; S1/M1.3 lessons are linked and navigable
2. "Coming soon" items show stub UI (not linked, descriptive message)
3. Information Theory module (5 lessons) renders fully: prose, KaTeX, code, inline quizzes
4. Quiz selection works: choose option → explanation revealed → persists across refresh
5. Progress persists: complete a lesson, reload — sidebar + dashboard reflect it
6. Dashboard shows correct resume lesson, streak, completion stats
7. `pnpm dev` from root starts both services with no manual steps
8. No console errors on lesson, dashboard, or semester pages
