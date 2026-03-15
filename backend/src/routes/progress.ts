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
