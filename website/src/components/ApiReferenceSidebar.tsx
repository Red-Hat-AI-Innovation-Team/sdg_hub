"use client";

import { useState, useEffect, useCallback } from "react";

interface NavEntry {
  name: string;
  id: string;
}

interface NavSubcategory {
  label: string;
  entries: NavEntry[];
}

interface NavCategory {
  label: string;
  subcategories: NavSubcategory[];
}

export function ApiReferenceSidebar({
  navigation,
}: {
  navigation: NavCategory[];
}) {
  const [activeId, setActiveId] = useState<string>("");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const toggleSection = (key: string) => {
    setCollapsed((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // Scroll tracking via IntersectionObserver
  const setupObserver = useCallback(() => {
    const headings = document.querySelectorAll("[data-api-class]");
    if (headings.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        // Find the topmost visible heading
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length > 0 && visible[0].target.id) {
          setActiveId(visible[0].target.id);
        }
      },
      { rootMargin: "-80px 0px -60% 0px", threshold: 0 }
    );

    headings.forEach((h) => observer.observe(h));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const cleanup = setupObserver();
    return cleanup;
  }, [setupObserver]);

  // Set initial active from hash
  useEffect(() => {
    if (window.location.hash) {
      setActiveId(window.location.hash.slice(1));
    }
  }, []);

  const handleClick = (id: string) => {
    setActiveId(id);
    setMobileOpen(false);
    const el = document.getElementById(id);
    if (el) {
      const top = el.getBoundingClientRect().top + window.scrollY - 80;
      window.scrollTo({ top, behavior: "smooth" });
    }
  };

  const sidebarContent = (
    <nav className="px-3 py-5">
      <div className="mb-4 px-2">
        <span className="text-xs font-bold uppercase tracking-widest text-text-muted">
          API Reference
        </span>
      </div>
      {navigation.map((category) => (
        <div key={category.label} className="mb-4">
          <button
            onClick={() => toggleSection(category.label)}
            className="flex w-full items-center justify-between px-2 py-1.5 text-[11px] font-bold uppercase tracking-wider text-text"
          >
            {category.label}
            <svg
              className={`h-3 w-3 transform text-text-muted transition-transform ${
                collapsed[category.label] ? "-rotate-90" : ""
              }`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>
          {!collapsed[category.label] && (
            <div className="mt-1">
              {category.subcategories.map((sub) => (
                <div key={sub.label} className="mb-1">
                  <span className="block px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
                    {sub.label}
                  </span>
                  <ul>
                    {sub.entries.map((entry) => {
                      const isActive = activeId === entry.id;
                      return (
                        <li key={entry.id}>
                          <button
                            onClick={() => handleClick(entry.id)}
                            className={`block w-full truncate rounded px-3 py-1 text-left font-mono text-[13px] transition-colors ${
                              isActive
                                ? "bg-accent/8 font-medium text-accent"
                                : "text-text-muted hover:bg-bg-subtle hover:text-text"
                            }`}
                          >
                            {entry.name}
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </nav>
  );

  return (
    <>
      {/* Mobile toggle */}
      <button
        onClick={() => setMobileOpen(!mobileOpen)}
        className="fixed bottom-4 right-4 z-50 flex h-12 w-12 items-center justify-center rounded-full bg-text text-bg shadow-lg lg:hidden"
        aria-label="Toggle API navigation"
      >
        <svg
          className="h-5 w-5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          {mobileOpen ? (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          ) : (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          )}
        </svg>
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/30 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed top-16 z-40 h-[calc(100vh-4rem)] w-[240px] overflow-y-auto border-r border-border bg-bg-sidebar transition-transform lg:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {sidebarContent}
      </aside>
    </>
  );
}
