import { Routes, Route, Navigate } from 'react-router-dom';
import { AppShell } from './components/layout/AppShell.tsx';
import { DashboardPage } from './pages/DashboardPage.tsx';
import SemesterPage from './pages/SemesterPage.tsx';
import LessonPage from './pages/LessonPage.tsx';

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<DashboardPage />} />
        <Route path="/semester/:semId" element={<SemesterPage />} />
        <Route path="/lesson/:semId/:modId/:lessonId" element={<LessonPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
