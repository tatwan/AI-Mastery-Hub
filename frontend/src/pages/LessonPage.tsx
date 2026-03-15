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
      <div className="max-w-4xl mx-auto px-8 py-12 animate-pulse space-y-4">
        <div className="h-3 bg-slate-800 rounded w-1/3" />
        <div className="h-8 bg-slate-800 rounded w-2/3" />
        <div className="h-3 bg-slate-800 rounded" />
        <div className="h-3 bg-slate-800 rounded w-5/6" />
      </div>
    );
  }

  if (isError || !lesson) {
    return (
      <div className="max-w-4xl mx-auto px-8 py-12">
        <p className="text-red-400 mb-4">Lesson not found.</p>
        <Link to="/" className="text-indigo-400 hover:text-indigo-300">← Back to dashboard</Link>
      </div>
    );
  }

  const lessonProgress = progress?.lessons[lessonId];
  const isCompleted = lessonProgress?.status === 'completed';
  const quizAnswers = lessonProgress?.quizAnswers ?? {};

  return (
    <article className="max-w-4xl mx-auto px-8 py-12">
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
