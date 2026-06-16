import ep from "../data/courses/engineering-philosophy.json";
import sa from "../data/courses/system-architecture.json";
import vc from "../data/courses/video-codec.json";
import jw from "../data/courses/jepa-world-models.json";
import { marked } from "marked";
import katex from "katex";

export interface Chapter {
  number: number;
  emoji: string;
  title: string;
  titleKr: string;
  tldr: string;
  topics?: string[];
  learningGoals?: string[];
  overview?: string;
  sections?: { title: string; content: string; figures?: { src: string; caption: string; label?: string; arxivId?: string }[] }[];
  analogy?: { title: string; content: string };
  codeExample?: { language?: string; intro?: string; code?: string; walkthrough?: string };
  industryEvaluation?: {
    overview?: string;
    whatEngineersLookFor?: string[];
    redFlags?: string[];
    interviewQuestions?: string[];
    masteryVsFamiliar?: string;
  };
  keyTakeaways?: { title: string; content: string }[];
}

export interface Course {
  topic: string;
  topicKr?: string;
  topicSlug: string;
  level?: string;
  codeLanguage?: string;
  description?: string;
  categorySlug?: string;
  heroImage?: string;        // optional banner shown above the syllabus
  heroImageAlt?: string;
  chapters: Chapter[];
}

export const COURSES: Record<string, Course> = {
  "engineering-philosophy": ep as Course,
  "system-architecture":   sa as Course,
  "video-codec":           vc as Course,
  "jepa-world-models":      jw as Course,
};

export function chapterSlug(n: number) {
  return `chapter-${String(n).padStart(2, "0")}`;
}

export function fmtChapterNo(n: number, total: number) {
  return `CHAPTER ${String(n).padStart(2, "0")} OF ${String(total).padStart(2, "0")}`;
}

// inline markdown: bold, code, italic — no block wrapping
export function md(input: string): string {
  if (!input) return "";
  return marked.parseInline(input) as string;
}

function renderMath(text: string): { result: string; slots: string[] } {
  const slots: string[] = [];
  const placeholder = (i: number) => `\x00MATH${i}\x00`;

  // block math $$...$$
  let result = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, expr) => {
    const html = katex.renderToString(expr.trim(), { displayMode: true, throwOnError: false });
    slots.push(html);
    return placeholder(slots.length - 1);
  });

  // inline math $...$  (not $$)
  result = result.replace(/\$([^$\n]+?)\$/g, (_, expr) => {
    const html = katex.renderToString(expr.trim(), { displayMode: false, throwOnError: false });
    slots.push(html);
    return placeholder(slots.length - 1);
  });

  return { result, slots };
}

function restoreMath(html: string, slots: string[]): string {
  return html.replace(/\x00MATH(\d+)\x00/g, (_, i) => slots[parseInt(i)]);
}

// Insert paragraph breaks before "**Term** = ..." definition-list patterns
function splitDefinitions(text: string): string {
  // Only applies when multiple bold-term definitions run together in one block
  return text.replace(/([.。])\s+(\*\*[^*\n]+\*\*\s*[=—:])/g, "$1\n\n$2");
}

// full markdown: paragraphs, tables, lists, code blocks, math, auto-links
export function paragraphs(text: string): string {
  if (!text) return "";
  const { result: mathReplaced, slots } = renderMath(text);
  const withBreaks = splitDefinitions(mathReplaced);
  const html = marked.parse(withBreaks) as string;
  return restoreMath(html, slots);
}
