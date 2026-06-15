import ep from "../data/courses/engineering-philosophy.json";
import sa from "../data/courses/system-architecture.json";
import vc from "../data/courses/video-codec.json";

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
};

export function chapterSlug(n: number) {
  return `chapter-${String(n).padStart(2, "0")}`;
}

export function fmtChapterNo(n: number, total: number) {
  return `CHAPTER ${String(n).padStart(2, "0")} OF ${String(total).padStart(2, "0")}`;
}

// minimal markdown for prose: **bold** and `code`
export function md(input: string): string {
  if (!input) return "";
  return input
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

// split prose into HTML paragraphs (handles \n\n)
export function paragraphs(text: string): string {
  if (!text) return "";
  const blocks = text.split(/\n\n+/).map((b) => b.trim()).filter(Boolean);
  return blocks.map((b) => `<p>${md(b)}</p>`).join("\n");
}
