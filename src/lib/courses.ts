import ep from "../data/courses/engineering-philosophy.json";
import sa from "../data/courses/system-architecture.json";
import vc from "../data/courses/video-codec.json";
import jw from "../data/courses/jepa-world-models.json";
import { marked } from "marked";

export interface Chapter {
  number: number;
  emoji: string;
  title: string;
  titleKr: string;
  tldr: string;
  topics?: string[];
  learningGoals?: string[];
  overview?: string;
  sections?: { title: string; content: string }[];
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

// full markdown: paragraphs, tables, lists, code blocks, etc.
export function paragraphs(text: string): string {
  if (!text) return "";
  return marked.parse(text) as string;
}
