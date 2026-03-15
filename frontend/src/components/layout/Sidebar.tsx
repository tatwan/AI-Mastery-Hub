import { Link, useParams } from 'react-router-dom';
import { useCurriculum } from '../../hooks/useCurriculum.ts';
import type { SemesterMeta, ModuleMeta, LessonMeta } from '../../types.ts';

function LessonItem({
  lesson,
  semId,
  modId,
  isActive,
  isCompleted,
}: {
  lesson: LessonMeta;
  semId: string;
  modId: string;
  isActive: boolean;
  isCompleted: boolean;
}) {
  const base = 'flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-colors pl-8';

  if (lesson.status === 'coming-soon') {
    return (
      <div className={`${base} text-slate-600 cursor-default`}>
        <span className="w-3 h-3 flex-shrink-0" />
        {lesson.title}
      </div>
    );
  }

  return (
    <Link
      to={`/lesson/${semId}/${modId}/${lesson.id}`}
      className={`${base} ${
        isActive
          ? 'bg-indigo-500/20 text-indigo-300 font-medium'
          : isCompleted
          ? 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
          : 'text-slate-300 hover:text-slate-100 hover:bg-white/5'
      }`}
    >
      {isCompleted ? (
        <span className="w-3 text-emerald-500 flex-shrink-0">✓</span>
      ) : isActive ? (
        <span className="w-3 h-3 rounded-full bg-indigo-500 flex-shrink-0 inline-block" />
      ) : (
        <span className="w-3 h-3 flex-shrink-0" />
      )}
      {lesson.title}
    </Link>
  );
}

function ModuleSection({
  mod,
  semId,
  activeLessonId,
  completedLessons,
}: {
  mod: ModuleMeta;
  semId: string;
  activeLessonId?: string;
  completedLessons: Set<string>;
}) {
  if (mod.status !== 'available') {
    return (
      <div className="flex items-center justify-between px-3 py-1.5 mb-1">
        <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider truncate">
          {mod.title}
        </span>
        <span className="text-[10px] text-slate-700 bg-slate-800 px-1.5 py-0.5 rounded flex-shrink-0 ml-2">
          soon
        </span>
      </div>
    );
  }

  return (
    <div className="mb-2">
      <div className="px-3 py-1 text-xs font-semibold text-slate-400 uppercase tracking-wider">
        {mod.title}
      </div>
      {mod.lessons.map(lesson => (
        <LessonItem
          key={lesson.id}
          lesson={lesson}
          semId={semId}
          modId={mod.id}
          isActive={lesson.id === activeLessonId}
          isCompleted={completedLessons.has(lesson.id)}
        />
      ))}
    </div>
  );
}

function SemesterSection({
  sem,
  activeLessonId,
  completedLessons,
}: {
  sem: SemesterMeta;
  activeLessonId?: string;
  completedLessons: Set<string>;
}) {
  return (
    <div className="mb-5">
      <div className="px-3 py-2 flex items-center justify-between">
        <span
          className={`text-sm font-bold ${
            sem.status === 'available' ? 'text-slate-200' : 'text-slate-600'
          }`}
        >
          {sem.title}
        </span>
        {sem.status !== 'available' && (
          <span className="text-[10px] text-slate-700 bg-slate-800/80 px-1.5 py-0.5 rounded flex-shrink-0 ml-2">
            coming soon
          </span>
        )}
      </div>
      {sem.status === 'available' &&
        sem.modules.map(mod => (
          <ModuleSection
            key={mod.id}
            mod={mod}
            semId={sem.id}
            activeLessonId={activeLessonId}
            completedLessons={completedLessons}
          />
        ))}
    </div>
  );
}

export function Sidebar({ completedLessons = new Set<string>() }: { completedLessons?: Set<string> }) {
  const { data: curriculum, isLoading } = useCurriculum();
  const params = useParams<{ lessonId?: string }>();

  return (
    <aside className="w-[280px] flex-shrink-0 min-h-0 bg-slate-900 border-r border-white/5 overflow-y-auto">
      <div className="pt-4 pb-8">
        {isLoading && (
          <div className="px-4 text-slate-600 text-sm animate-pulse">Loading…</div>
        )}
        {curriculum?.semesters.map(sem => (
          <SemesterSection
            key={sem.id}
            sem={sem}
            activeLessonId={params.lessonId}
            completedLessons={completedLessons}
          />
        ))}
      </div>
    </aside>
  );
}
