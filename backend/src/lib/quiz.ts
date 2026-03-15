import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkDirective from 'remark-directive';
import remarkFrontmatter from 'remark-frontmatter';
import { visit } from 'unist-util-visit';
import yaml from 'js-yaml';
import type { Node } from 'unist';

export interface MarkdownBlock {
  type: 'markdown';
  raw: string;
}

export interface QuizBlock {
  type: 'quiz';
  question: string;
  options: string[];
  correct: number;
  explanation: string;
}

export type ContentBlock = MarkdownBlock | QuizBlock;

interface RawQuizFields {
  question: unknown;
  options: unknown;
  correct: unknown;
  explanation: unknown;
}

// Position data on remark nodes
interface NodeWithPosition extends Node {
  name?: string;
  position?: {
    start: { offset?: number };
    end: { offset?: number };
  };
}

function isValidQuiz(
  parsed: RawQuizFields
): parsed is { question: string; options: string[]; correct: number; explanation: string } {
  return (
    typeof parsed.question === 'string' &&
    Array.isArray(parsed.options) &&
    (parsed.options as unknown[]).length >= 2 &&
    typeof parsed.correct === 'number' &&
    typeof parsed.explanation === 'string'
  );
}

// The processor is reused across calls — unified processors are safe to reuse
// (each .parse() call returns a new tree; the processor is not mutated).
const processor = unified()
  .use(remarkParse)
  .use(remarkFrontmatter)
  .use(remarkDirective);

export function splitContent(markdown: string): ContentBlock[] {
  const tree = processor.parse(markdown);

  // Collect quiz directive positions in document order.
  // We use source-position offsets to slice YAML directly from the raw markdown string.
  // This avoids deep AST text traversal (remark parses YAML list values as listItem nodes,
  // not flat text children — AST text extraction would silently drop option values).
  const quizPositions: Array<{ start: number; end: number }> = [];

  visit(tree, 'containerDirective', (node: Node) => {
    const n = node as NodeWithPosition;
    if (n.name === 'quiz' && n.position) {
      quizPositions.push({
        start: n.position.start.offset ?? 0,
        end: n.position.end.offset ?? 0,
      });
    }
  });

  const blocks: ContentBlock[] = [];
  let lastIndex = 0;

  for (const { start, end } of quizPositions) {
    const preceding = markdown.slice(lastIndex, start).trim();
    if (preceding) {
      blocks.push({ type: 'markdown', raw: preceding });
    }

    // Extract YAML body by slicing between ":::quiz\n" and "\n:::"
    // Using the source string directly is more reliable than walking the AST.
    const blockSource = markdown.slice(start, end);
    const bodyMatch = blockSource.match(/^:::quiz\n([\s\S]*?)\n:::$/);
    const yamlText = bodyMatch ? bodyMatch[1] : '';

    try {
      const parsed = yaml.load(yamlText) as RawQuizFields;
      if (isValidQuiz(parsed)) {
        blocks.push({
          type: 'quiz',
          question: parsed.question,
          options: parsed.options,
          correct: parsed.correct,
          explanation: parsed.explanation,
        });
      } else {
        console.warn('Quiz block missing required fields — skipping');
      }
    } catch (err) {
      console.warn('Failed to parse quiz YAML — skipping:', err);
    }

    lastIndex = end;
  }

  const remaining = markdown.slice(lastIndex).trim();
  if (remaining) {
    blocks.push({ type: 'markdown', raw: remaining });
  }

  return blocks;
}
