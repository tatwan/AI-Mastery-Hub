import { useEffect, useRef, useState } from "react";
import hljs from "highlight.js";
import { Play, Copy, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import "highlight.js/styles/atom-one-dark.css";

interface CodeBlockProps {
  code: string;
  language?: string;
  runnable?: boolean;
}

export function CodeBlock({ code, language = "python", runnable = true }: CodeBlockProps) {
  const codeRef = useRef<HTMLElement>(null);
  const [copied, setCopied] = useState(false);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    if (codeRef.current) {
      delete codeRef.current.dataset.highlighted;
      hljs.highlightElement(codeRef.current);
    }
  }, [code, language]);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRun = () => {
    setIsRunning(true);
    // Simulate execution delay
    setTimeout(() => {
      setIsRunning(false);
    }, 800);
  };

  return (
    <div className="relative group my-6 rounded-xl overflow-hidden bg-[#282c34] border border-white/10 shadow-xl">
      <div className="flex items-center justify-between px-4 py-2 border-b border-white/10 bg-black/40">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/80" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
            <div className="w-3 h-3 rounded-full bg-green-500/80" />
          </div>
          <span className="ml-2 text-xs font-mono text-muted-foreground uppercase">{language}</span>
        </div>
        <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <button 
            onClick={handleCopy}
            className="p-1.5 rounded-md hover:bg-white/10 text-muted-foreground hover:text-foreground transition-colors"
            title="Copy code"
          >
            {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
          </button>
          {runnable && (
            <button 
              onClick={handleRun}
              disabled={isRunning}
              className={cn(
                "flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-semibold transition-all",
                isRunning ? "bg-primary/50 text-primary-foreground/50 cursor-not-allowed" : "bg-primary text-primary-foreground hover:bg-primary/90"
              )}
            >
              <Play className="w-3 h-3" />
              {isRunning ? "Running..." : "Run"}
            </button>
          )}
        </div>
      </div>
      <div className="p-4 overflow-x-auto text-sm font-mono leading-relaxed">
        <pre className="!m-0 !p-0 bg-transparent">
          <code ref={codeRef} className={`language-${language} !bg-transparent`}>
            {code}
          </code>
        </pre>
      </div>
    </div>
  );
}
