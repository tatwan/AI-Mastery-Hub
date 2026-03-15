import type { ProgressData } from '../../types.ts';

export function StatsRow({ stats }: { stats: ProgressData['stats'] }) {
  const items = [
    { label: 'Day Streak', value: stats.streak > 0 ? `${stats.streak} 🔥` : stats.streak },
    { label: 'Lessons Done', value: stats.totalCompleted },
    {
      label: 'Last Active',
      value: stats.lastActivityAt
        ? new Date(stats.lastActivityAt).toLocaleDateString()
        : '—',
    },
    { label: 'XP Earned', value: stats.totalCompleted * 70 },
  ];

  return (
    <div className="grid grid-cols-2 gap-4">
      {items.map(({ label, value }) => (
        <div key={label} className="p-4 rounded-xl bg-slate-800/50 border border-white/5">
          <p className="text-xs text-slate-500 mb-1">{label}</p>
          <p className="text-xl font-extrabold text-slate-100">{value}</p>
        </div>
      ))}
    </div>
  );
}
