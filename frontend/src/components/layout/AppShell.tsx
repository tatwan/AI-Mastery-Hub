import { Outlet } from 'react-router-dom';
import { Header } from './Header.tsx';
import { Sidebar } from './Sidebar.tsx';
import { ErrorBoundary } from '../ErrorBoundary.tsx';
import { FirstVisitToast } from '../Toast.tsx';
import { useProgress } from '../../hooks/useProgress.ts';

export function AppShell() {
  const { data: progress } = useProgress();
  const completedLessons = new Set(
    Object.entries(progress?.lessons ?? {})
      .filter(([, lp]) => lp.status === 'completed')
      .map(([id]) => id)
  );

  return (
    <div className="flex flex-col h-screen">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar completedLessons={completedLessons} />
        <main className="flex-1 min-h-0 overflow-y-auto bg-surface-deep">
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
      <FirstVisitToast />
    </div>
  );
}
