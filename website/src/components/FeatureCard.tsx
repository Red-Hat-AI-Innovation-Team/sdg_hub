"use client";

import Link from "next/link";
import { type ReactNode } from "react";

interface FeatureCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  href: string;
  glowColor: string;
}

export function FeatureCard({ icon, title, description, href, glowColor }: FeatureCardProps) {
  return (
    <Link
      href={href}
      className="group cursor-pointer rounded-xl p-6 transition-all duration-200 no-underline"
      style={{
        background: "var(--color-bg-2)",
        boxShadow: "0 0 0 1px var(--color-border)",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-2px)";
        e.currentTarget.style.boxShadow = `0 0 0 1px var(--color-border-strong), 0 4px 20px ${glowColor}`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.boxShadow = "0 0 0 1px var(--color-border)";
      }}
    >
      <div className="mb-3 text-accent">{icon}</div>
      <h3
        className="text-lg font-semibold text-text-0"
        style={{ fontFamily: "var(--font-heading)" }}
      >
        {title}
      </h3>
      <p className="mt-2 text-sm leading-relaxed text-text-2">
        {description}
      </p>
      <span className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-text-3 opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100">
        Learn more
        <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
        </svg>
      </span>
    </Link>
  );
}
