import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import type { Components } from 'react-markdown';
import { CodeBlock } from './CodeBlock.tsx';
import { InlineQuiz } from './InlineQuiz.tsx';
import type { ContentBlock, Lesson } from '../../types.ts';

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
    return (
      <blockquote className="border-l-4 border-indigo-500 pl-4 my-6 text-slate-300 bg-indigo-500/5 py-3 rounded-r-lg">
        {children}
      </blockquote>
    );
  },
};

interface Props {
  lesson: Lesson;
  quizAnswers: Record<string, number>;
  onQuizAnswer: (quizIndex: number, selectedOption: number) => void;
}

export function LessonReader({ lesson, quizAnswers, onQuizAnswer }: Props) {
  // quizCount tracks index among quiz blocks only (not the full content array index)
  // so quizAnswers keys "0","1",... always refer to the Nth quiz in the lesson
  let quizCount = 0;

  return (
    <div>
      {lesson.content.map((block: ContentBlock, i: number) => {
        if (block.type === 'markdown') {
          return (
            <ReactMarkdown
              key={i}
              remarkPlugins={[remarkMath]}
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
