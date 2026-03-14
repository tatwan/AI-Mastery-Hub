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
