"use client";

const pipelineSteps = [
  { label: "dataset", color: "var(--color-text-2)" },
  { label: "PromptBuilder", color: "var(--color-accent)" },
  { label: "LLMChat", color: "var(--color-green)" },
  { label: "TagParser", color: "var(--color-blue)" },
  { label: "Filter", color: "var(--color-purple)" },
  { label: "enriched", color: "var(--color-text-2)" },
];

export function AnimatedPipeline() {
  return (
    <div
      className="relative mt-16 flex flex-wrap items-center justify-center gap-y-4"
      style={{ fontFamily: "var(--font-mono)" }}
    >
      {pipelineSteps.map((step, i) => (
        <span key={step.label} className="flex items-center">
          {/* Pipeline node */}
          <span
            className="pipeline-node rounded-lg px-3 py-1.5 text-xs sm:text-sm"
            style={
              {
                background: "var(--color-bg-2)",
                color: step.color,
                boxShadow: "0 0 0 1px var(--color-border)",
                "--glow-color": step.color,
                animationDelay: `${i * 0.7}s`,
              } as React.CSSProperties
            }
          >
            {step.label}
          </span>

          {/* Connector with animated dot */}
          {i < pipelineSteps.length - 1 && (
            <span className="connector-wrapper mx-1 hidden sm:flex sm:mx-2">
              <span className="connector-line" />
              <span
                className="data-packet"
                style={{ animationDelay: `${i * 0.7}s` }}
              />
            </span>
          )}
        </span>
      ))}
    </div>
  );
}
