import Link from "next/link";
import { Header } from "@/components/Header";
import { TypewriterSubtitle } from "@/components/TypewriterSubtitle";
import { AnimatedPipeline } from "@/components/AnimatedPipeline";
import { FeatureCard } from "@/components/FeatureCard";

const features = [
  {
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M14.25 6.087c0-.355.186-.676.401-.959.221-.29.349-.634.349-1.003 0-1.036-1.007-1.875-2.25-1.875s-2.25.84-2.25 1.875c0 .369.128.713.349 1.003.215.283.401.604.401.959v0a.64.64 0 01-.657.643 48.39 48.39 0 01-4.163-.3c.186 1.613.293 3.25.315 4.907a.656.656 0 01-.658.663v0c-.355 0-.676-.186-.959-.401a1.647 1.647 0 00-1.003-.349c-1.036 0-1.875 1.007-1.875 2.25s.84 2.25 1.875 2.25c.369 0 .713-.128 1.003-.349.283-.215.604-.401.959-.401v0c.31 0 .555.26.532.57a48.039 48.039 0 01-.642 5.056c1.518.19 3.058.309 4.616.354a.64.64 0 00.657-.643v0c0-.355-.186-.676-.401-.959a1.647 1.647 0 01-.349-1.003c0-1.035 1.008-1.875 2.25-1.875 1.243 0 2.25.84 2.25 1.875 0 .369-.128.713-.349 1.003-.215.283-.401.604-.401.959v0c0 .333.277.599.61.58a48.1 48.1 0 005.427-.63 48.05 48.05 0 00.582-4.717.532.532 0 00-.533-.57v0c-.355 0-.676.186-.959.401-.29.221-.634.349-1.003.349-1.035 0-1.875-1.007-1.875-2.25s.84-2.25 1.875-2.25c.37 0 .713.128 1.003.349.283.215.604.401.959.401v0a.656.656 0 00.658-.663 48.422 48.422 0 00-.37-5.36c-1.886.342-3.81.574-5.766.689a.578.578 0 01-.61-.58v0z" />
      </svg>
    ),
    title: "Composable Blocks",
    description:
      "Chain LLM, parsing, transform, filtering, and agent blocks in any order. Each block does one thing well.",
    href: "/docs/blocks",
    glowColor: "rgba(232, 151, 93, 0.15)",
  },
  {
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
      </svg>
    ),
    title: "YAML Flows",
    description:
      "Define multi-step pipelines in YAML. Portable, reproducible, and version-controlled by design.",
    href: "/docs/flows",
    glowColor: "rgba(128, 151, 196, 0.15)",
  },
  {
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
      </svg>
    ),
    title: "Auto-Discovery",
    description:
      "BlockRegistry and FlowRegistry find and catalog all available components automatically. Zero boilerplate.",
    href: "/docs/concepts",
    glowColor: "rgba(168, 139, 184, 0.15)",
  },
  {
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
      </svg>
    ),
    title: "Async Performance",
    description:
      "100+ LLM providers through LiteLLM with async execution. Built for throughput from the ground up.",
    href: "/docs/blocks/llm-blocks",
    glowColor: "rgba(125, 170, 140, 0.15)",
  },
];

export default function Home() {
  return (
    <div className="flex min-h-screen w-full flex-col overflow-x-hidden" style={{ background: "var(--color-bg-0)" }}>
      <Header />

      {/* Hero */}
      <section className="relative flex w-full flex-1 flex-col items-center justify-center px-4 py-28 text-center sm:px-6">
        {/* Amber glow background */}
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 60% 40% at 50% 30%, rgba(232, 151, 93, 0.06) 0%, transparent 70%)",
          }}
        />

        {/* Version badge */}
        <div
          className="relative mb-8 inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-medium text-text-2"
          style={{
            background: "var(--color-bg-2)",
            boxShadow: "0 0 0 1px var(--color-border-strong)",
          }}
        >
          <span
            className="pulse-dot inline-block h-2 w-2 rounded-full"
            style={{ background: "var(--color-green)" }}
          />
          Open Source
        </div>

        {/* Title */}
        <h1
          className="relative max-w-4xl text-4xl font-bold leading-tight tracking-tight sm:text-5xl lg:text-6xl"
          style={{
            fontFamily: "var(--font-heading)",
            color: "var(--color-text-0)",
          }}
        >
          Build synthetic data pipelines
          <br />
          with <span className="shimmer-text">composable blocks</span>
        </h1>

        {/* Typewriter subtitle */}
        <TypewriterSubtitle />

        {/* CTA buttons */}
        <div className="relative mt-10 flex flex-wrap justify-center gap-4">
          <Link
            href="/docs"
            className="rounded-lg px-6 py-2.5 text-sm font-medium text-bg-0 transition-all hover:brightness-110"
            style={{ background: "var(--color-accent)" }}
          >
            Get Started
          </Link>
          <Link
            href="/api-reference"
            className="rounded-lg px-6 py-2.5 text-sm font-medium text-text-1 transition-colors hover:text-text-0"
            style={{
              background: "var(--color-bg-3)",
              boxShadow: "0 0 0 1px var(--color-border-strong)",
            }}
          >
            API Reference
          </Link>
        </div>

        {/* Pipeline visualization */}
        <AnimatedPipeline />
      </section>

      {/* Features */}
      <section
        className="w-full px-4 py-20 sm:px-6"
        style={{
          borderTop: "1px solid var(--color-border)",
          background: "var(--color-bg-1)",
        }}
      >
        <div className="mx-auto grid w-full max-w-4xl gap-6 sm:grid-cols-2">
          {features.map((feature) => (
            <FeatureCard
              key={feature.title}
              icon={feature.icon}
              title={feature.title}
              description={feature.description}
              href={feature.href}
              glowColor={feature.glowColor}
            />
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer
        className="px-6 py-8 text-center text-sm text-text-3"
        style={{ borderTop: "1px solid var(--color-border)" }}
      >
        Built by Red Hat AI Innovation Team
      </footer>
    </div>
  );
}
