import { Link } from 'react-router-dom';
import type { Curriculum, ProgressData } from '../../types.ts';

interface ResumeTarget {
  semId: string; modId: string; lessonId: string; title: string;
  moduleName: string; semesterName: string; completionPct: number;
}

function findResumeTarget(curriculum: Curriculum, progress: ProgressData): ResumeTarget | null {
  for (const sem of curriculum.semesters) {
    if (sem.status !== 'available') continue;
    for (const mod of sem.modules) {
      if (mod.status !== 'available') continue;
      for (const lesson of mod.lessons) {
        if (lesson.status !== 'available') continue;
        const lp = progress.lessons[lesson.id];
        if (!lp || lp.status !== 'completed') {
          const availableLessons = mod.lessons.filter(l => l.status === 'available');
          const done = availableLessons.filter(l => progress.lessons[l.id]?.status === 'completed').length;
          return {
            semId: sem.id, modId: mod.id, lessonId: lesson.id, title: lesson.title,
            moduleName: mod.title, semesterName: sem.title,
            completionPct: availableLessons.length > 0
              ? Math.round((done / availableLessons.length) * 100)
              : 0,
          };
        }
      }
    }
  }
  return null;
}

export function ResumeCard({ curriculum, progress }: { curriculum: Curriculum; progress: ProgressData }) {
  const target = findResumeTarget(curriculum, progress);

  if (!target) {
    const hasAvailable = curriculum.semesters.some(sem =>
      sem.status === 'available' && sem.modules.some(mod =>
        mod.status === 'available' && mod.lessons.some(l => l.status === 'available')
      )
    );

    return (
      <div className="p-6 rounded-xl bg-slate-800/50 border border-white/8 text-center">
        {hasAvailable ? (
          <>
            <span className="text-emerald-400 font-bold">All available lessons complete!</span>
            <p className="text-slate-400 text-sm mt-1">More content coming soon.</p>
          </>
        ) : (
          <>
            <span className="text-slate-300 font-bold">No lessons available yet.</span>
            <p className="text-slate-400 text-sm mt-1">Check back soon.</p>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="p-6 rounded-xl bg-gradient-to-br from-indigo-500/10 to-violet-500/10 border border-indigo-500/20">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <p className="text-xs text-slate-400 mb-1">{target.semesterName} · {target.moduleName}</p>
          <h2 className="text-lg font-bold text-slate-100 truncate">{target.title}</h2>
          <div className="mt-3 flex items-center gap-3">
            <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 rounded-full transition-all"
                style={{ width: `${target.completionPct}%` }}
              />
            </div>
            <span className="text-xs text-slate-400 flex-shrink-0">{target.completionPct}%</span>
          </div>
        </div>
        <Link
          to={`/lesson/${target.semId}/${target.modId}/${target.lessonId}`}
          className="flex-shrink-0 px-5 py-2.5 rounded-lg bg-gradient-to-r from-indigo-500 to-violet-500 text-white font-semibold text-sm hover:opacity-90 transition-opacity"
        >
          Resume
        </Link>
      </div>
    </div>
  );
}
