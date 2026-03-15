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
  // selectedOption === -1 signals "clear this answer" (quiz reset)
  if (selectedOption === -1) {
    delete answers[String(quizIndex)];
  } else {
    answers[String(quizIndex)] = selectedOption;
  }
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
