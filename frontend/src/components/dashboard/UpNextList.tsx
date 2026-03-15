import { Link } from 'react-router-dom';
import type { Curriculum, ProgressData } from '../../types.ts';

export function UpNextList({ curriculum, progress }: { curriculum: Curriculum; progress: ProgressData }) {
  const upNext: Array<{ semId: string; modId: string; lessonId: string; title: string }> = [];
  let skippedFirst = false;

  outer: for (const sem of curriculum.semesters) {
    if (sem.status !== 'available') continue;
    for (const mod of sem.modules) {
      if (mod.status !== 'available') continue;
      for (const lesson of mod.lessons) {
        if (lesson.status !== 'available') continue;
        const lp = progress.lessons[lesson.id];
        if (!lp || lp.status !== 'completed') {
          if (!skippedFirst) { skippedFirst = true; continue; }
          upNext.push({ semId: sem.id, modId: mod.id, lessonId: lesson.id, title: lesson.title });
          if (upNext.length >= 3) break outer;
        }
      }
    }
  }

  if (!upNext.length) return null;

  return (
    <div className="p-5 rounded-xl bg-slate-800/50 border border-white/5">
      <h3 className="text-sm font-semibold text-slate-400 mb-3">Up Next</h3>
      <ul className="space-y-2">
        {upNext.map(item => (
          <li key={item.lessonId}>
            <Link
              to={`/lesson/${item.semId}/${item.modId}/${item.lessonId}`}
              className="text-sm text-slate-300 hover:text-indigo-300 transition-colors flex items-center gap-2"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-500/60 flex-shrink-0" />
              {item.title}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
