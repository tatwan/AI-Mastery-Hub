import { useState } from "react";
import { useSubmitExercise } from "@workspace/api-client-react";
import { Exercise } from "@workspace/api-client-react/src/generated/api.schemas";
import { CodeBlock } from "./CodeBlock";
import { CheckCircle2, XCircle, AlertCircle, ChevronRight, HelpCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface ExercisePanelProps {
  exercise: Exercise;
  onSuccess?: () => void;
}

export function ExercisePanel({ exercise, onSuccess }: ExercisePanelProps) {
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [codeAnswer, setCodeAnswer] = useState<string>(exercise.starterCode || "");
  const [showHint, setShowHint] = useState(false);
  const [result, setResult] = useState<{ isCorrect: boolean; explanation: string; xpEarned: number } | null>(null);
  
  const submitMutation = useSubmitExercise();

  const handleSubmit = async () => {
    let answer = "";
    if (exercise.type === "multiple_choice" || exercise.type === "true_false") {
      answer = selectedOption || "";
    } else {
      answer = codeAnswer;
    }

    if (!answer) return;

    try {
      const res = await submitMutation.mutateAsync({
        exerciseId: exercise.id,
        data: { answer, code: codeAnswer }
      });
      
      setResult({
        isCorrect: res.isCorrect,
        explanation: res.explanation,
        xpEarned: res.xpEarned
      });

      if (res.isCorrect && onSuccess) {
        onSuccess();
      }
    } catch (error) {
      console.error("Failed to submit", error);
    }
  };

  return (
    <div className="bg-card border border-border/50 rounded-2xl overflow-hidden shadow-2xl flex flex-col h-full">
      <div className="p-4 bg-muted/30 border-b border-border/50 flex items-center justify-between">
        <h3 className="font-semibold flex items-center gap-2">
          <span className="w-6 h-6 rounded-md bg-primary/20 text-primary flex items-center justify-center text-sm">
            Q
          </span>
          Knowledge Check
        </h3>
        {result && result.isCorrect && (
          <span className="flex items-center gap-1 text-xs font-medium text-green-400 bg-green-400/10 px-2 py-1 rounded-full">
            <CheckCircle2 className="w-3 h-3" /> +{result.xpEarned} XP
          </span>
        )}
      </div>

      <div className="p-6 flex-1 overflow-y-auto space-y-6">
        <div className="prose-academic text-base">
          <p className="font-medium text-foreground">{exercise.question}</p>
          {exercise.description && <p className="text-muted-foreground text-sm mt-2">{exercise.description}</p>}
        </div>

        {/* Multiple Choice & True/False */}
        {(exercise.type === "multiple_choice" || exercise.type === "true_false") && (
          <div className="space-y-3">
            {exercise.options?.map((opt, i) => {
              const isSelected = selectedOption === opt;
              const isSubmitted = result !== null;
              const isCorrectAnswer = result?.isCorrect && isSelected;
              const isWrongAnswer = !result?.isCorrect && isSelected && isSubmitted;

              return (
                <button
                  key={i}
                  disabled={isSubmitted && result?.isCorrect}
                  onClick={() => !result?.isCorrect && setSelectedOption(opt)}
                  className={cn(
                    "w-full text-left p-4 rounded-xl border transition-all duration-200 flex items-start gap-3",
                    isSelected && !isSubmitted ? "border-primary bg-primary/5 shadow-[0_0_0_1px_rgba(99,102,241,0.5)]" : "border-border hover:border-border/80 bg-background/50 hover:bg-muted/50",
                    isCorrectAnswer ? "border-green-500/50 bg-green-500/10" : "",
                    isWrongAnswer ? "border-red-500/50 bg-red-500/10" : "",
                    isSubmitted && result.isCorrect && !isSelected ? "opacity-50 cursor-not-allowed" : ""
                  )}
                >
                  <div className={cn(
                    "w-5 h-5 rounded-full border flex-shrink-0 flex items-center justify-center mt-0.5",
                    isSelected ? "border-primary" : "border-muted-foreground",
                    isCorrectAnswer ? "border-green-500 bg-green-500 text-white" : "",
                    isWrongAnswer ? "border-red-500" : ""
                  )}>
                    {isCorrectAnswer && <CheckCircle2 className="w-3 h-3" />}
                    {isWrongAnswer && <XCircle className="w-3 h-3 text-red-500" />}
                    {isSelected && !isSubmitted && <div className="w-2.5 h-2.5 rounded-full bg-primary" />}
                  </div>
                  <span className="text-sm font-medium leading-tight">{opt}</span>
                </button>
              );
            })}
          </div>
        )}

        {/* Code Exercises */}
        {(exercise.type === "write_code" || exercise.type === "fill_code") && (
          <div className="space-y-4">
            <div className="rounded-xl overflow-hidden border border-border focus-within:border-primary focus-within:ring-1 focus-within:ring-primary transition-all">
              <textarea
                value={codeAnswer}
                onChange={(e) => setCodeAnswer(e.target.value)}
                disabled={result?.isCorrect}
                className="w-full h-48 bg-[#1e1e24] p-4 text-sm font-mono text-gray-300 resize-y focus:outline-none"
                placeholder="# Write your Python solution here..."
                spellCheck={false}
              />
            </div>
          </div>
        )}

        {/* Hints */}
        {exercise.hints && exercise.hints.length > 0 && !result?.isCorrect && (
          <div className="mt-6 border-t border-border/50 pt-4">
            <button 
              onClick={() => setShowHint(!showHint)}
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <HelpCircle className="w-4 h-4" />
              {showHint ? "Hide hint" : "Stuck? View a hint"}
            </button>
            {showHint && (
              <div className="mt-3 p-3 rounded-lg bg-accent/5 border border-accent/20 text-sm text-accent-foreground/90">
                <ul className="list-disc pl-5 space-y-1">
                  {exercise.hints.map((hint, i) => <li key={i}>{hint}</li>)}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className={cn(
            "p-4 rounded-xl border mt-6 animate-in fade-in slide-in-from-bottom-2",
            result.isCorrect ? "bg-green-500/10 border-green-500/30 text-green-200" : "bg-red-500/10 border-red-500/30 text-red-200"
          )}>
            <div className="flex gap-3">
              {result.isCorrect ? <CheckCircle2 className="w-5 h-5 text-green-400 shrink-0" /> : <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />}
              <div>
                <h4 className="font-semibold mb-1">
                  {result.isCorrect ? "Excellent work!" : "Not quite right."}
                </h4>
                <p className="text-sm opacity-90">{result.explanation}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="p-4 border-t border-border/50 bg-card/50 backdrop-blur">
        <button
          onClick={handleSubmit}
          disabled={submitMutation.isPending || (exercise.type !== "write_code" && !selectedOption) || result?.isCorrect}
          className="w-full py-3 px-4 rounded-xl font-semibold bg-gradient-to-r from-primary to-primary/80 text-white shadow-lg shadow-primary/20 hover:shadow-xl hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center justify-center gap-2"
        >
          {submitMutation.isPending ? "Checking..." : result?.isCorrect ? "Completed" : "Submit Answer"}
          {!result?.isCorrect && !submitMutation.isPending && <ChevronRight className="w-4 h-4" />}
        </button>
      </div>
    </div>
  );
}
