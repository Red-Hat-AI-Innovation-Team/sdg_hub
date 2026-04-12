export interface NavItem {
  title: string;
  href: string;
}

export interface NavSection {
  title: string;
  items: NavItem[];
}

export const navigation: NavSection[] = [
  {
    title: "Getting Started",
    items: [
      { title: "Introduction", href: "/docs" },
      { title: "Installation", href: "/docs/installation" },
      { title: "Quick Start", href: "/docs/quickstart" },
      { title: "Core Concepts", href: "/docs/concepts" },
    ],
  },
  {
    title: "Blocks",
    items: [
      { title: "Overview", href: "/docs/blocks" },
      { title: "LLM Blocks", href: "/docs/blocks/llm-blocks" },
      { title: "Parsing Blocks", href: "/docs/blocks/parsing-blocks" },
      { title: "Transform Blocks", href: "/docs/blocks/transform-blocks" },
      { title: "Filtering Blocks", href: "/docs/blocks/filtering-blocks" },
      { title: "Agent Blocks", href: "/docs/blocks/agent-blocks" },
      { title: "Custom Blocks", href: "/docs/blocks/custom-blocks" },
    ],
  },
  {
    title: "Flows",
    items: [
      { title: "Overview", href: "/docs/flows" },
      { title: "YAML Reference", href: "/docs/flows/yaml-reference" },
      { title: "Built-in Flows", href: "/docs/flows/built-in-flows" },
      { title: "Custom Flows", href: "/docs/flows/custom-flows" },
    ],
  },
  {
    title: "Connectors",
    items: [{ title: "Overview", href: "/docs/connectors" }],
  },
  {
    title: "Reference",
    items: [
      { title: "Overview", href: "/docs/reference" },
      { title: "Blocks API", href: "/docs/reference/blocks" },
      { title: "Flow API", href: "/docs/reference/flow" },
      { title: "Connectors API", href: "/docs/reference/connectors" },
    ],
  },
  {
    title: "Contributing",
    items: [{ title: "Development", href: "/docs/development" }],
  },
];

/**
 * Returns a flat ordered list of all nav items, used for prev/next links.
 */
export function getFlatNavItems(): NavItem[] {
  return navigation.flatMap((section) => section.items);
}

/**
 * Find prev/next pages relative to a given href.
 */
export function getPrevNext(href: string): {
  prev: NavItem | null;
  next: NavItem | null;
} {
  const flat = getFlatNavItems();
  const index = flat.findIndex((item) => item.href === href);

  if (index === -1) {
    return { prev: null, next: null };
  }

  return {
    prev: index > 0 ? flat[index - 1] : null,
    next: index < flat.length - 1 ? flat[index + 1] : null,
  };
}
