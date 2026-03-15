import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useState } from 'react';

interface Props {
  language?: string;
  children: string;
}

export function CodeBlock({ language = 'python', children }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    void navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-6 rounded-lg overflow-hidden border border-white/8">
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-white/8">
        <span className="text-xs text-slate-500 font-mono">{language}</span>
        <button
          onClick={handleCopy}
          className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        customStyle={{ margin: 0, padding: '1rem', background: 'rgb(15, 23, 42)', fontSize: '0.875rem' }}
        PreTag="div"
      >
        {children}
      </SyntaxHighlighter>
    </div>
  );
}
