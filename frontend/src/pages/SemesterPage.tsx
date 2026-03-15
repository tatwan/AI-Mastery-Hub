import { useParams } from 'react-router-dom';

export default function SemesterPage() {
  const { semId } = useParams();
  return (
    <div className="max-w-4xl mx-auto px-8 py-12">
      <p className="text-slate-500 text-sm font-mono">{semId}</p>
      <h1 className="text-2xl font-extrabold tracking-tight mt-1">Coming Soon</h1>
      <p className="text-slate-400 mt-3">This semester is under construction.</p>
    </div>
  );
}
