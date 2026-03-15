import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/**
 * Renders inline text that may contain $math$ expressions.
 * Strips the surrounding <p> tag so it works inline in buttons/labels.
 */
export function MathText({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        p: ({ children: inner }) => <>{inner}</>,
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
