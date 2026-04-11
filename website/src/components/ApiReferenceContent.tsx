"use client";

import { useEffect } from "react";

// ── Types (mirrored from page for client use) ─────────────────────

interface FieldDef {
  name: string;
  type: string;
  required: boolean;
  default: string | number | boolean | null;
  description: string;
}

interface ParameterDef {
  name: string;
  type?: string;
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

interface ClassSection {
  categoryLabel: string;
  subcategoryLabel: string;
  cls: ClassDef;
}

// ── Helpers ───────────────────────────────────────────────────────

/** Trim docstrings: take only the first paragraph (before a blank line or "Parameters" / "Attributes" heading). */
function summaryFromDocstring(ds: string): string {
  if (!ds) return "";
  const lines = ds.split("\n");
  const out: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (
      trimmed === "" ||
      trimmed.startsWith("Parameters") ||
      trimmed.startsWith("Attributes") ||
      trimmed.startsWith("Returns") ||
      trimmed.startsWith("Args:")
    ) {
      break;
    }
    out.push(trimmed);
  }
  return out.join(" ");
}

/** Format a default value for display. */
function formatDefault(val: string | number | boolean | null): string {
  if (val === null) return "--";
  if (typeof val === "string") {
    if (val === "") return '""';
    return `"${val}"`;
  }
  if (typeof val === "boolean") return val ? "True" : "False";
  return String(val);
}

/** Shorten long type annotations for display. */
function shortType(t: string | undefined | null): string {
  if (!t) return "";
  return t
    .replace(/typing\./g, "")
    .replace(/sdg_hub\.core\.\w+\.\w+\./g, "")
    .replace(/pandas\.core\.frame\./g, "")
    .replace(/datasets\.arrow_dataset\./g, "")
    .replace(/Union\[([^,]+), NoneType\]/g, "Optional[$1]");
}

// ── Copy button ───────────────────────────────────────────────────

function CopyButton({ text }: { text: string }) {
  const handleCopy = () => {
    navigator.clipboard.writeText(text);
  };
  return (
    <button
      onClick={handleCopy}
      className="ml-2 inline-flex shrink-0 items-center rounded px-1.5 py-0.5 text-[11px] text-text-muted transition-colors hover:bg-border hover:text-text"
      title="Copy import"
    >
      <svg className="mr-1 h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
        />
      </svg>
      Copy
    </button>
  );
}

// ── Badge components ──────────────────────────────────────────────

function Badge({ children, variant }: { children: React.ReactNode; variant: "abstract" | "classmethod" | "static" }) {
  const colors = {
    abstract: "bg-amber-50 text-amber-700 border-amber-200",
    classmethod: "bg-indigo-50 text-indigo-700 border-indigo-200",
    static: "bg-emerald-50 text-emerald-700 border-emerald-200",
  };
  return (
    <span className={`inline-block rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${colors[variant]}`}>
      {children}
    </span>
  );
}

// ── Fields table ──────────────────────────────────────────────────

function FieldsTable({ fields }: { fields: FieldDef[] }) {
  if (fields.length === 0) return null;
  return (
    <div className="api-section">
      <h4 className="api-section-heading">Fields</h4>
      <div className="overflow-x-auto">
        <table className="api-table">
          <thead>
            <tr>
              <th className="w-[180px]">Name</th>
              <th className="w-[240px]">Type</th>
              <th className="w-[100px]">Default</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {fields.map((f) => (
              <tr key={f.name}>
                <td>
                  <code className="api-field-name">{f.name}</code>
                  {f.required && <span className="api-required-dot" title="Required" />}
                </td>
                <td>
                  <code className="api-type">{shortType(f.type)}</code>
                </td>
                <td className="api-default">
                  {f.required ? (
                    <span className="text-[11px] font-medium uppercase tracking-wider text-accent">required</span>
                  ) : (
                    <code>{formatDefault(f.default)}</code>
                  )}
                </td>
                <td className="text-text-muted">{f.description || "--"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Method block ──────────────────────────────────────────────────

function MethodBlock({ method, className }: { method: MethodDef; className: string }) {
  const docSummary = summaryFromDocstring(method.docstring);
  const hasParams = method.parameters.length > 0;

  return (
    <div className="api-method">
      <div className="api-method-header">
        <div className="flex flex-wrap items-center gap-2">
          <code className="api-method-name">{className}.{method.name}</code>
          {method.abstract && <Badge variant="abstract">abstract</Badge>}
          {method.classmethod && <Badge variant="classmethod">classmethod</Badge>}
          {method.staticmethod && <Badge variant="static">static</Badge>}
        </div>
      </div>
      <div className="api-signature">
        <code>{shortType(method.signature)}</code>
      </div>
      {docSummary && <p className="api-method-doc">{docSummary}</p>}
      {hasParams && (
        <div className="mt-3">
          <table className="api-table api-table-compact">
            <thead>
              <tr>
                <th className="w-[160px]">Parameter</th>
                <th className="w-[220px]">Type</th>
                <th>Default</th>
              </tr>
            </thead>
            <tbody>
              {method.parameters.map((p) => (
                <tr key={p.name}>
                  <td>
                    <code className="api-field-name">{p.name}</code>
                  </td>
                  <td>
                    <code className="api-type">{shortType(p.type)}</code>
                  </td>
                  <td className="api-default">
                    {"default" in p ? (
                      <code>{formatDefault(p.default ?? null)}</code>
                    ) : (
                      <span className="text-[11px] font-medium uppercase tracking-wider text-accent">required</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {method.return_type && (
        <div className="mt-2 text-[13px]">
          <span className="font-medium text-text-muted">Returns </span>
          <code className="api-type">{shortType(method.return_type)}</code>
        </div>
      )}
    </div>
  );
}

// ── Single class section ──────────────────────────────────────────

function ClassBlock({ section }: { section: ClassSection }) {
  const { cls, categoryLabel, subcategoryLabel } = section;
  const docSummary = summaryFromDocstring(cls.docstring);
  const hasMethods = cls.methods.length > 0;

  return (
    <section id={cls.name} data-api-class className="api-class-section">
      {/* Breadcrumb */}
      <div className="api-breadcrumb">
        {categoryLabel} / {subcategoryLabel}
      </div>

      {/* Class heading */}
      <h2 className="api-class-name">
        <a href={`#${cls.name}`} className="api-class-anchor">
          {cls.name}
        </a>
      </h2>

      {/* Bases */}
      {cls.bases.length > 0 && (
        <div className="mb-3 text-[13px] text-text-muted">
          Bases: {cls.bases.map((b, i) => (
            <span key={b}>
              {i > 0 && ", "}
              <code className="api-type">{b}</code>
            </span>
          ))}
        </div>
      )}

      {/* Import */}
      <div className="api-import-block">
        <code>{cls.import_path}</code>
        <CopyButton text={cls.import_path} />
      </div>

      {/* Docstring */}
      {docSummary && <p className="api-docstring">{docSummary}</p>}

      {/* Fields */}
      <FieldsTable fields={cls.fields} />

      {/* Methods */}
      {hasMethods && (
        <div className="api-section">
          <h4 className="api-section-heading">Methods</h4>
          <div className="space-y-4">
            {cls.methods.map((m) => (
              <MethodBlock key={m.name} method={m} className={cls.name} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

// ── Main content component ────────────────────────────────────────

export function ApiReferenceContent({
  sections,
}: {
  sections: ClassSection[];
}) {
  // Scroll to hash on mount
  useEffect(() => {
    if (window.location.hash) {
      const id = window.location.hash.slice(1);
      const el = document.getElementById(id);
      if (el) {
        setTimeout(() => {
          const top = el.getBoundingClientRect().top + window.scrollY - 80;
          window.scrollTo({ top, behavior: "smooth" });
        }, 100);
      }
    }
  }, []);

  return (
    <div className="api-reference-content">
      {/* Page header */}
      <div className="mb-10 border-b border-border pb-8">
        <h1 className="font-serif text-3xl font-bold tracking-tight text-text">
          API Reference
        </h1>
        <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-text-muted">
          Complete reference for all public classes in SDG Hub. Every block, flow component, and connector is documented with its fields, methods, and type signatures.
        </p>
      </div>

      {/* Class sections */}
      {sections.map((section) => (
        <ClassBlock key={section.cls.name} section={section} />
      ))}
    </div>
  );
}
