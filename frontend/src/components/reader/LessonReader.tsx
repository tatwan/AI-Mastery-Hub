import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import type { Components } from 'react-markdown';
import { CodeBlock } from './CodeBlock.tsx';
import { InlineQuiz } from './InlineQuiz.tsx';
import type { ContentBlock, Lesson } from '../../types.ts';

function extractText(node: React.ReactNode): string {
  if (typeof node === 'string') return node;
  if (typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(extractText).join('');
  if (React.isValidElement(node)) return extractText((node.props as { children?: React.ReactNode }).children);
  return '';
}

const mdComponents: Components = {
  code({ className, children }) {
    const match = /language-(\w+)/.exec(className ?? '');
    const code = String(children).replace(/\n$/, '');
    if (match) return <CodeBlock language={match[1]}>{code}</CodeBlock>;
    return (
      <code className="bg-slate-800 text-indigo-300 px-1.5 py-0.5 rounded text-sm font-mono">
        {children}
      </code>
    );
  },
  h2({ children }) {
    return (
      <h2 className="text-xl font-extrabold tracking-tight text-slate-100 mt-10 mb-4 border-b border-white/5 pb-2">
        {children}
      </h2>
    );
  },
  h3({ children }) {
    return <h3 className="text-lg font-bold text-slate-200 mt-8 mb-3">{children}</h3>;
  },
  p({ children }) {
    return <p className="text-slate-300 leading-relaxed mb-4">{children}</p>;
  },
  ul({ children }) {
    return <ul className="list-disc pl-6 text-slate-300 mb-4 space-y-1">{children}</ul>;
  },
  ol({ children }) {
    return <ol className="list-decimal pl-6 text-slate-300 mb-4 space-y-1">{children}</ol>;
  },
  blockquote({ children }) {
    // Detect callout type by keyword prefix in the rendered text.
    // Content convention: > **Refresher:** | > **Intuition:** | > **Remember:** | > **Key insight:**
    const text = extractText(children).trim();

    if (text.startsWith('Refresher:')) {
      return (
        <blockquote className="border-l-4 border-orange-500/70 pl-4 my-6 text-slate-300 bg-orange-500/5 py-3 rounded-r-lg">
          {children}
        </blockquote>
      );
    }
    if (text.startsWith('Intuition:')) {
      return (
        <blockquote className="border-l-4 border-emerald-500/70 pl-4 my-6 text-slate-300 bg-emerald-500/5 py-3 rounded-r-lg">
          {children}
        </blockquote>
      );
    }
    if (text.startsWith('Remember:') || text.startsWith('Note:')) {
      return (
        <blockquote className="border-l-4 border-yellow-500/70 pl-4 my-6 text-slate-300 bg-yellow-500/5 py-3 rounded-r-lg">
          {children}
        </blockquote>
      );
    }
    // Default: Key insight / general callouts → indigo
    return (
      <blockquote className="border-l-4 border-indigo-500 pl-4 my-6 text-slate-300 bg-indigo-500/5 py-3 rounded-r-lg">
        {children}
      </blockquote>
    );
  },
  table({ children }) {
    return (
      <div className="my-6 overflow-x-auto">
        <table className="w-full text-sm border-collapse">{children}</table>
      </div>
    );
  },
  thead({ children }) {
    return <thead className="border-b border-white/10">{children}</thead>;
  },
  tbody({ children }) {
    return <tbody className="divide-y divide-white/5">{children}</tbody>;
  },
  tr({ children }) {
    return <tr className="hover:bg-white/2 transition-colors">{children}</tr>;
  },
  th({ children }) {
    return (
      <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
        {children}
      </th>
    );
  },
  td({ children }) {
    return <td className="px-4 py-2.5 text-slate-300">{children}</td>;
  },
};

interface Props {
  lesson: Lesson;
  quizAnswers: Record<string, number>;
  onQuizAnswer: (quizIndex: number, selectedOption: number) => void;
}

export function LessonReader({ lesson, quizAnswers, onQuizAnswer }: Props) {
  let quizCount = 0;

  return (
    <div>
      {lesson.content.map((block: ContentBlock, i: number) => {
        if (block.type === 'markdown') {
          return (
            <ReactMarkdown
              key={i}
              remarkPlugins={[remarkMath, remarkGfm]}
              rehypePlugins={[rehypeKatex]}
              components={mdComponents}
            >
              {block.raw}
            </ReactMarkdown>
          );
        }
        const qIdx = quizCount++;
        return (
          <InlineQuiz
            key={i}
            quizIndex={qIdx}
            block={block}
            savedAnswer={quizAnswers[String(qIdx)]}
            onAnswer={selected => onQuizAnswer(qIdx, selected)}
          />
        );
      })}
    </div>
  );
}
