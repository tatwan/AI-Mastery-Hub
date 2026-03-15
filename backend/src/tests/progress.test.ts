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
