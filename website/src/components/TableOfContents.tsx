"use client";

import { useEffect, useState } from "react";

interface Heading {
  id: string;
  text: string;
  level: number;
}

export function TableOfContents({ headings }: { headings: Heading[] }) {
  const [activeId, setActiveId] = useState<string>("");

  useEffect(() => {
    if (headings.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        // Find the first heading that is visible
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
            break;
          }
        }
      },
      { rootMargin: "-80px 0px -60% 0px", threshold: 0 }
    );

    for (const heading of headings) {
      const el = document.getElementById(heading.id);
      if (el) observer.observe(el);
    }

    return () => observer.disconnect();
  }, [headings]);

  if (headings.length === 0) return null;

  return (
    <nav className="hidden xl:block">
      <div className="sticky top-24">
        <p
          className="mb-3 text-[11px] tracking-wider text-text-3"
          style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}
        >
          on_this_page
        </p>
        <ul
          className="space-y-1.5 pl-3 text-sm"
          style={{ borderLeft: "1px solid var(--color-border)" }}
        >
          {headings.map((heading) => (
            <li key={heading.id}>
              <a
                href={`#${heading.id}`}
                className={`block transition-colors ${
                  heading.level === 3 ? "pl-3" : ""
                } ${
                  activeId === heading.id
                    ? "font-medium text-accent"
                    : "text-text-3 hover:text-text-1"
                }`}
              >
                {heading.text}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
