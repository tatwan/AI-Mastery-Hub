import { Link } from 'react-router-dom';

export function Header() {
  return (
    <header className="h-12 bg-slate-900 border-b border-white/5 flex items-center px-6 flex-shrink-0 z-10">
      <Link to="/" className="flex items-center gap-2">
        <div className="w-5 h-5 rounded bg-gradient-to-br from-indigo-500 to-violet-500 flex-shrink-0" />
        <span className="font-extrabold tracking-tight text-slate-100 text-sm">
          AI Mastery Hub
        </span>
      </Link>
    </header>
  );
}
