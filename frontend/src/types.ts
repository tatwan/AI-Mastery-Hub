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
