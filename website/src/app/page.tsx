import Link from "next/link";
import { Header } from "@/components/Header";

const features = [
  {
    title: "Composable Blocks",
    description:
      "Chain LLM, parsing, transform, filtering, and agent blocks in any order. Each block does one thing well.",
  },
  {
    title: "YAML Flows",
    description:
      "Define multi-step pipelines in YAML. Portable, reproducible, and version-controlled by design.",
  },
  {
    title: "Auto-Discovery",
    description:
      "BlockRegistry and FlowRegistry find and catalog all available components automatically. Zero boilerplate.",
  },
  {
    title: "Async Performance",
    description:
      "100+ LLM providers through LiteLLM with async execution. Built for throughput from the ground up.",
  },
];

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      {/* Hero */}
      <section className="flex flex-1 flex-col items-center justify-center px-6 py-24 text-center">
        <h1 className="font-serif text-5xl font-bold leading-tight tracking-tight text-text sm:text-6xl">
          SDG Hub
        </h1>
        <p className="mt-5 max-w-xl text-lg leading-relaxed text-text-muted">
          A modular Python framework for building synthetic data generation
          pipelines using composable blocks and flows.
        </p>
        <div className="mt-8 flex gap-4">
          <Link
            href="/docs"
            className="rounded-lg bg-accent px-6 py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
          >
            Get Started
          </Link>
          <Link
            href="/docs/reference"
            className="rounded-lg border border-border px-6 py-2.5 text-sm font-medium text-text transition-colors hover:border-text-muted"
          >
            API Reference
          </Link>
        </div>

        {/* Pipeline visual */}
        <div className="mt-14 flex items-center gap-3 font-mono text-sm text-text-muted">
          <span className="rounded bg-bg-subtle px-3 py-1.5">dataset</span>
          <span>&rarr;</span>
          <span className="rounded bg-bg-subtle px-3 py-1.5">Block 1</span>
          <span>&rarr;</span>
          <span className="rounded bg-bg-subtle px-3 py-1.5">Block 2</span>
          <span>&rarr;</span>
          <span className="rounded bg-bg-subtle px-3 py-1.5">Block 3</span>
          <span>&rarr;</span>
          <span className="rounded bg-bg-subtle px-3 py-1.5">
            enriched_dataset
          </span>
        </div>
      </section>

      {/* Features */}
      <section className="border-t border-border bg-bg-subtle px-6 py-20">
        <div className="mx-auto grid max-w-4xl gap-8 sm:grid-cols-2">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="rounded-lg border border-border bg-bg p-6"
            >
              <h3 className="font-serif text-lg font-semibold text-text">
                {feature.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-text-muted">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border px-6 py-8 text-center text-sm text-text-muted">
        SDG Hub &mdash; Red Hat AI Innovation Team
      </footer>
    </div>
  );
}
