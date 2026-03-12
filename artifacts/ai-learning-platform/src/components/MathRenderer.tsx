import { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

interface MathRendererProps {
  content: string;
  block?: boolean;
}

export function MathRenderer({ content, block = false }: MathRendererProps) {
  const containerRef = useRef<HTMLDivElement | HTMLSpanElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      try {
        katex.render(content, containerRef.current, {
          displayMode: block,
          throwOnError: false,
          strict: false,
        });
      } catch (err) {
        console.error("KaTeX rendering error:", err);
        containerRef.current.textContent = content;
      }
    }
  }, [content, block]);

  if (block) {
    return <div ref={containerRef as any} className="my-6 w-full flex justify-center text-lg" />;
  }
  
  return <span ref={containerRef as any} className="inline-block px-1" />;
}
