import { useEffect, useState } from 'react';
import { isFirstVisit, markVisited } from '../lib/session.ts';

export function FirstVisitToast() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (isFirstVisit()) {
      markVisited();
      setVisible(true);
      const t = setTimeout(() => setVisible(false), 6000);
      return () => clearTimeout(t);
    }
  }, []);

  if (!visible) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 max-w-sm p-4 rounded-xl bg-slate-800 border border-indigo-500/30 shadow-2xl">
      <p className="text-sm font-semibold text-slate-200">Welcome to AI Mastery Hub</p>
      <p className="text-xs text-slate-400 mt-1">
        Your progress is saved locally in this browser. No account needed.
      </p>
    </div>
  );
}
