import { describe, it, expect } from 'vitest';
import { splitContent } from '../lib/quiz.js';

const SAMPLE_MD = `## Overview

Some prose here with math.

:::quiz
question: "What maximizes entropy?"
options:
  - "Uniform distribution"
  - "Skewed distribution"
correct: 0
explanation: "The uniform distribution maximizes entropy."
:::

More prose after the quiz.`;

describe('splitContent', () => {
  it('splits markdown into prose and quiz blocks', () => {
    const blocks = splitContent(SAMPLE_MD);
    expect(blocks.length).toBe(3);
    expect(blocks[0].type).toBe('markdown');
    expect(blocks[1].type).toBe('quiz');
    expect(blocks[2].type).toBe('markdown');
  });

  it('quiz block has correct fields', () => {
    const blocks = splitContent(SAMPLE_MD);
    const quiz = blocks[1];
    if (quiz.type !== 'quiz') throw new Error('expected quiz block');
    expect(quiz.question).toBe('What maximizes entropy?');
    expect(quiz.options).toHaveLength(2);
    expect(quiz.correct).toBe(0);
    expect(quiz.explanation).toBeTruthy();
  });

  it('returns single markdown block if no quiz directives', () => {
    const blocks = splitContent('Just prose here.');
    expect(blocks).toHaveLength(1);
    expect(blocks[0].type).toBe('markdown');
  });

  it('omits quiz blocks with missing required fields', () => {
    const md = `Prose.\n\n:::quiz\nquestion: "Missing correct field"\noptions:\n  - "A"\n:::\n\nMore.`;
    const blocks = splitContent(md);
    expect(blocks.every(b => b.type === 'markdown')).toBe(true);
  });
});
