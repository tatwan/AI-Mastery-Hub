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
