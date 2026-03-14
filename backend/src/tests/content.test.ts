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
