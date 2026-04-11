import fs from "fs";
import path from "path";
import matter from "gray-matter";
import Markdoc, { type RenderableTreeNode } from "@markdoc/markdoc";
import { markdocConfig, resetHeadingIds, highlightCode } from "./markdoc";

const DOCS_DIR = path.join(/* turbopackIgnore: true */ process.cwd(), "..", "docs");

/**
 * Preprocess markdown source to strip MkDocs-specific syntax that Markdoc
 * does not understand.
 *
 * Handles the `=== "Tab Title"` tabbed content syntax by:
 *   1. Removing the `=== "..."` header lines entirely
 *   2. Dedenting the content that was nested under each tab by one level
 *      (4 spaces) so that fenced code blocks are recognised normally
 */
function preprocessMarkdown(source: string): string {
  const lines = source.split("\n");
  const result: string[] = [];
  let insideTab = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Match MkDocs tab header: === "title"
    if (/^===\s+"[^"]*"\s*$/.test(line)) {
      insideTab = true;
      continue; // drop the === line
    }

    if (insideTab) {
      // A non-empty line that is NOT indented (and is not blank) signals the
      // end of the tabbed block.
      if (line.length > 0 && !line.startsWith("    ") && !line.startsWith("\t")) {
        insideTab = false;
        result.push(line);
      } else {
        // Dedent by 4 spaces (one MkDocs tab-indent level)
        result.push(line.startsWith("    ") ? line.slice(4) : line);
      }
    } else {
      result.push(line);
    }
  }

  return result.join("\n");
}

export interface DocPage {
  title: string;
  description?: string;
  content: RenderableTreeNode;
  htmlContent: string;
  slug: string[];
}

/**
 * Render Markdoc AST to HTML string with syntax-highlighted code blocks.
 */
function renderToHighlightedHtml(content: RenderableTreeNode): string {
  // Render to HTML string first
  let html = Markdoc.renderers.html(content);

  // Replace <pre data-language="..."><code ...>...</code></pre> with highlighted versions
  html = html.replace(
    /<pre data-language="([^"]*)"[^>]*><code[^>]*>([\s\S]*?)<\/code><\/pre>/g,
    (_match, lang, code) => {
      // Decode HTML entities back to raw text for the highlighter
      const decoded = code
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/&amp;/g, "&")
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'");
      return highlightCode(decoded.trim(), lang);
    }
  );

  return html;
}

/**
 * Map a URL slug array to a file path in the docs directory.
 *
 *   []                        -> docs/index.md
 *   ["installation"]          -> docs/installation.md
 *   ["blocks", "llm-blocks"]  -> docs/blocks/llm-blocks.md
 *   ["blocks"]                -> docs/blocks/index.md
 */
function slugToFilePath(slug: string[]): string | null {
  if (slug.length === 0) {
    const p = path.join(DOCS_DIR, "index.md");
    return fs.existsSync(p) ? p : null;
  }

  // Try direct file first, e.g. docs/installation.md
  const directFile = path.join(DOCS_DIR, ...slug) + ".md";
  if (fs.existsSync(directFile)) return directFile;

  // Try index inside directory, e.g. docs/blocks/index.md
  const indexFile = path.join(DOCS_DIR, ...slug, "index.md");
  if (fs.existsSync(indexFile)) return indexFile;

  return null;
}

/**
 * Extract a title from the first `# Heading` in the markdown source.
 */
function extractTitle(source: string): string {
  const match = source.match(/^#\s+(.+)$/m);
  return match ? match[1] : "Untitled";
}

/**
 * Load and parse a single doc page by its URL slug segments.
 */
export function getDocPage(slug: string[]): DocPage | null {
  const filePath = slugToFilePath(slug);
  if (!filePath) return null;

  const raw = fs.readFileSync(filePath, "utf-8");
  const { content: source, data: frontmatter } = matter(raw);

  const preprocessed = preprocessMarkdown(source);
  resetHeadingIds();
  const ast = Markdoc.parse(preprocessed);
  const content = Markdoc.transform(ast, markdocConfig);

  const htmlContent = renderToHighlightedHtml(content);

  return {
    title: (frontmatter.title as string) || extractTitle(source),
    description: frontmatter.description as string | undefined,
    content,
    htmlContent,
    slug,
  };
}

/**
 * Collect all valid doc slugs for generateStaticParams.
 */
export function getAllDocSlugs(): string[][] {
  const slugs: string[][] = [];

  function walk(dir: string, segments: string[]) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        walk(path.join(dir, entry.name), [...segments, entry.name]);
      } else if (entry.name.endsWith(".md")) {
        if (entry.name === "index.md") {
          slugs.push(segments);
        } else {
          slugs.push([...segments, entry.name.replace(/\.md$/, "")]);
        }
      }
    }
  }

  walk(DOCS_DIR, []);
  return slugs;
}

/**
 * Extract headings (h2, h3) from a markdown source string for TOC generation.
 */
export function extractHeadings(
  slug: string[]
): { id: string; text: string; level: number }[] {
  const filePath = slugToFilePath(slug);
  if (!filePath) return [];

  const raw = fs.readFileSync(filePath, "utf-8");
  const { content: source } = matter(raw);
  const preprocessed = preprocessMarkdown(source);
  const headings: { id: string; text: string; level: number }[] = [];
  const regex = /^(#{2,3})\s+(.+)$/gm;
  const seenIds = new Map<string, number>();
  let match;

  while ((match = regex.exec(preprocessed)) !== null) {
    const level = match[1].length;
    const text = match[2].trim();
    const baseId = text
      .toLowerCase()
      .replace(/[^\w\s-]/g, "")
      .replace(/\s+/g, "-");
    const count = seenIds.get(baseId) || 0;
    seenIds.set(baseId, count + 1);
    const id = count === 0 ? baseId : `${baseId}-${count}`;
    headings.push({ id, text, level });
  }

  return headings;
}
