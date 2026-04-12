import { type Config, nodes, Tag } from "@markdoc/markdoc";
import { createHighlighterCoreSync } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import python from "shiki/langs/python.mjs";
import bash from "shiki/langs/bash.mjs";
import yaml from "shiki/langs/yaml.mjs";
import json from "shiki/langs/json.mjs";
import toml from "shiki/langs/toml.mjs";

const shiki = createHighlighterCoreSync({
  themes: [],
  langs: [python, bash, yaml, json, toml],
  engine: createJavaScriptRegexEngine(),
});

// Kanagawa-warm inspired theme
const kanagawa = {
  name: "kanagawa",
  type: "dark" as const,
  colors: {
    "editor.background": "#13111a",
    "editor.foreground": "#c8bfb0",
  },
  tokenColors: [
    { scope: ["comment", "punctuation.definition.comment"], settings: { foreground: "#3d3a36", fontStyle: "italic" } },
    { scope: ["string", "string.quoted"], settings: { foreground: "#8ea383" } },
    { scope: ["keyword", "storage.type", "storage.modifier"], settings: { foreground: "#c4746e" } },
    { scope: ["entity.name.function", "support.function", "meta.function-call"], settings: { foreground: "#7e9cba" } },
    { scope: ["variable", "variable.other", "variable.parameter"], settings: { foreground: "#c8bfb0" } },
    { scope: ["constant.numeric", "constant.language"], settings: { foreground: "#d4956a" } },
    { scope: ["entity.name.type", "support.type", "support.class", "entity.name.class"], settings: { foreground: "#a88bb8" } },
    { scope: ["keyword.operator"], settings: { foreground: "#a09b93" } },
    { scope: ["punctuation"], settings: { foreground: "#a09b93" } },
    { scope: ["entity.name.tag"], settings: { foreground: "#c4746e" } },
    { scope: ["entity.other.attribute-name"], settings: { foreground: "#d4956a" } },
    { scope: ["meta.decorator", "punctuation.decorator"], settings: { foreground: "#e8975d" } },
    { scope: ["keyword.control.import", "keyword.control.from"], settings: { foreground: "#c4746e" } },
  ],
};

/**
 * Highlight code with shiki using the kanagawa theme.
 * Returns raw HTML string with <pre><code> wrapping.
 */
export function highlightCode(code: string, lang: string): string {
  const supported = ["python", "bash", "yaml", "json", "toml", "sh", "shell"];
  const normalizedLang = lang === "sh" || lang === "shell" ? "bash" : lang;

  if (!supported.includes(lang) && lang !== "sh" && lang !== "shell") {
    // Return plain text for unsupported languages
    const escaped = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const safeLang = lang.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;"); return `<pre data-language="${safeLang}"><code>${escaped}</code></pre>`;
  }

  const html = shiki.codeToHtml(code, {
    lang: normalizedLang,
    theme: kanagawa,
  });

  // Inject our data-language attribute into the <pre> tag
  const safeLang = lang.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;"); return html.replace("<pre", `<pre data-language="${safeLang}"`);
}

/**
 * Generate a URL-friendly ID from heading text.
 * Uses a page-level counter to ensure uniqueness for duplicate headings.
 */
let headingCounter = 0;

function generateId(children: unknown[]): string {
  headingCounter++;
  const base = children
    .filter((child) => typeof child === "string")
    .join("")
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-");

  return `${base}-${headingCounter}`;
}

/** Reset heading counter between pages. */
export function resetHeadingIds() {
  headingCounter = 0;
}

export const markdocConfig: Config = {
  nodes: {
    heading: {
      ...nodes.heading,
      transform(node, config) {
        const attributes = node.transformAttributes(config);
        const children = node.transformChildren(config);
        const id = generateId(children);

        return new Tag(
          `h${node.attributes.level}`,
          { ...attributes, id },
          children
        );
      },
    },
    fence: {
      ...nodes.fence,
      transform(node, config) {
        const attributes = node.transformAttributes(config);
        const language = node.attributes.language || "";

        return new Tag(
          "pre",
          { "data-language": language, ...attributes },
          [
            new Tag("code", { class: language ? `language-${language}` : "" }, [
              node.attributes.content,
            ]),
          ]
        );
      },
    },
    link: {
      ...nodes.link,
      transform(node, config) {
        const attributes = node.transformAttributes(config);
        const children = node.transformChildren(config);
        let href = (node.attributes.href as string) || "";

        // Rewrite relative .md links to website paths (skip absolute URLs)
        if (href.endsWith(".md") && !href.startsWith("http")) {
          href = href
            .replace(/\.md$/, "")
            .replace(/\/index$/, "")
            .replace(/^\.\.\//, "")
            .replace(/^\.\//, "");

          // Make it an absolute /docs/ path
          if (!href.startsWith("/")) {
            href = `/docs/${href}`;
          }
        }

        return new Tag("a", { ...attributes, href }, children);
      },
    },
  },
  tags: {
    callout: {
      render: "Callout",
      attributes: {
        type: {
          type: String,
          default: "info",
          matches: ["info", "warning", "error"],
        },
      },
    },
  },
};
