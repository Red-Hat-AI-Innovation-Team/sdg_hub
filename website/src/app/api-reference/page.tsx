import fs from "fs";
import path from "path";
import { ApiReferenceSidebar } from "@/components/ApiReferenceSidebar";
import { ApiReferenceContent } from "@/components/ApiReferenceContent";

// ── Types ──────────────────────────────────────────────────────────

interface FieldDef {
  name: string;
  type: string;
  required: boolean;
  default: string | number | boolean | null;
  description: string;
}

interface ParameterDef {
  name: string;
  type: string;
  default?: string | number | boolean | null;
}

interface MethodDef {
  name: string;
  signature: string;
  docstring: string;
  abstract: boolean;
  classmethod: boolean;
  staticmethod: boolean;
  parameters: ParameterDef[];
  return_type: string;
}

interface ClassDef {
  name: string;
  module: string;
  import_path: string;
  docstring: string;
  bases: string[];
  fields: FieldDef[];
  methods: MethodDef[];
}

interface ApiData {
  [category: string]: {
    [subcategory: string]: ClassDef[];
  };
}

// ── Label helpers ──────────────────────────────────────────────────

const categoryLabels: Record<string, string> = {
  blocks: "Blocks",
  flow: "Flow",
  connectors: "Connectors",
};

const subcategoryLabels: Record<string, string> = {
  base: "Base",
  registry: "Registry",
  llm: "LLM",
  parsing: "Parsing",
  transform: "Transform",
  filtering: "Filtering",
  agent: "Agent",
  mcp: "MCP",
  metadata: "Metadata",
};

function labelFor(key: string, map: Record<string, string>): string {
  return map[key] ?? key.charAt(0).toUpperCase() + key.slice(1);
}

// ── Build sidebar navigation from API data ─────────────────────────

function buildNavigation(data: ApiData) {
  return Object.entries(data).map(([catKey, subs]) => ({
    label: labelFor(catKey, categoryLabels),
    subcategories: Object.entries(subs).map(([subKey, classes]) => ({
      label: labelFor(subKey, subcategoryLabels),
      entries: classes.map((cls) => ({
        name: cls.name,
        id: cls.name,
      })),
    })),
  }));
}

// ── Flatten classes for rendering ──────────────────────────────────

interface ClassSection {
  categoryLabel: string;
  subcategoryLabel: string;
  cls: ClassDef;
}

function flattenClasses(data: ApiData): ClassSection[] {
  const sections: ClassSection[] = [];
  for (const [catKey, subs] of Object.entries(data)) {
    for (const [subKey, classes] of Object.entries(subs)) {
      for (const cls of classes) {
        sections.push({
          categoryLabel: labelFor(catKey, categoryLabels),
          subcategoryLabel: labelFor(subKey, subcategoryLabels),
          cls,
        });
      }
    }
  }
  return sections;
}

// ── Page ───────────────────────────────────────────────────────────

export default function ApiReferencePage() {
  const jsonPath = path.join(process.cwd(), "public", "api-reference.json");
  const raw = fs.readFileSync(jsonPath, "utf-8");
  const data: ApiData = JSON.parse(raw);

  const navigation = buildNavigation(data);
  const sections = flattenClasses(data);

  return (
    <>
      <ApiReferenceSidebar navigation={navigation} />
      <main className="min-w-0 flex-1 overflow-y-auto lg:pl-[248px]">
        <ApiReferenceContent sections={sections} />
      </main>
    </>
  );
}
