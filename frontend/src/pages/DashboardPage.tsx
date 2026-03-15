import { useCurriculum } from '../hooks/useCurriculum.ts';
import { useProgress } from '../hooks/useProgress.ts';
import { ResumeCard } from '../components/dashboard/ResumeCard.tsx';
import { StatsRow } from '../components/dashboard/StatsRow.tsx';
import { UpNextList } from '../components/dashboard/UpNextList.tsx';

export function DashboardPage() {
  const { data: curriculum, isLoading: loadingCurr } = useCurriculum();
  const { data: progress, isLoading: loadingProg } = useProgress();

  if (loadingCurr || loadingProg) {
    return (
      <div className="max-w-4xl mx-auto px-8 py-12 animate-pulse space-y-6">
        <div className="h-6 bg-slate-800 rounded w-1/4" />
        <div className="h-28 bg-slate-800 rounded" />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-40 bg-slate-800 rounded" />
          <div className="h-40 bg-slate-800 rounded" />
        </div>
      </div>
    );
  }

  if (!curriculum || !progress) return null;

  return (
    <div className="max-w-4xl mx-auto px-8 py-12 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-extrabold tracking-tight text-slate-100">Welcome back</h1>
        {progress.stats.streak > 0 && (
          <span className="text-sm font-bold text-amber-400">{progress.stats.streak} 🔥</span>
        )}
      </div>

      <ResumeCard curriculum={curriculum} progress={progress} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <StatsRow stats={progress.stats} />
        <UpNextList curriculum={curriculum} progress={progress} />
      </div>
    </div>
  );
}
