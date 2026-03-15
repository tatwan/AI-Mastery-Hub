import { useState } from 'react';
import type { QuizBlock } from '../../types.ts';

interface Props {
  quizIndex: number;
  block: QuizBlock;
  savedAnswer?: number;
  onAnswer: (selectedOption: number) => void;
}

export function InlineQuiz({ block, savedAnswer, onAnswer }: Props) {
  const [selected, setSelected] = useState<number | undefined>(savedAnswer);
  const answered = selected !== undefined;
  const isCorrect = selected === block.correct;

  const handleSelect = (idx: number) => {
    if (answered) return;
    setSelected(idx);
    onAnswer(idx);
  };

  return (
    <div className="my-8 p-6 rounded-xl bg-slate-800/60 border border-white/8">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-5 h-5 rounded bg-indigo-500/20 flex items-center justify-center flex-shrink-0">
          <span className="text-indigo-400 text-xs font-bold">?</span>
        </div>
        <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
          Knowledge Check
        </span>
      </div>

      <p className="text-slate-200 font-medium mb-4">{block.question}</p>

      <div className="space-y-2">
        {block.options.map((option, idx) => {
          let style = 'border-white/10 text-slate-300 hover:border-indigo-500/50 hover:bg-indigo-500/5';
          if (answered) {
            if (idx === block.correct) style = 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300';
            else if (idx === selected) style = 'border-red-500/50 bg-red-500/10 text-red-300';
            else style = 'border-white/5 text-slate-600';
          }

          return (
            <button
              key={idx}
              onClick={() => handleSelect(idx)}
              disabled={answered}
              className={`w-full text-left px-4 py-3 rounded-lg border text-sm transition-all ${style} ${answered ? 'cursor-default' : 'cursor-pointer'}`}
            >
              <span className="font-mono text-xs mr-3 opacity-60">
                {String.fromCharCode(65 + idx)}.
              </span>
              {option}
            </button>
          );
        })}
      </div>

      {answered && (
        <div
          className={`mt-4 p-4 rounded-lg text-sm ${
            isCorrect
              ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-300'
              : 'bg-amber-500/10 border border-amber-500/20 text-amber-300'
          }`}
        >
          <span className="font-semibold mr-1">{isCorrect ? 'Correct!' : 'Not quite.'}</span>
          {block.explanation}
        </div>
      )}
    </div>
  );
}
