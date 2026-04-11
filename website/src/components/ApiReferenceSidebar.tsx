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

/** Convert "Base" to "base" for snake_case display */
function toSnakeLabel(str: string): string {
  return str.toLowerCase().replace(/\s+/g, "_");
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
        <span
          className="text-[11px] tracking-wider text-text-2"
          style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}
        >
          api_reference
        </span>
      </div>
      {navigation.map((category) => (
        <div key={category.label} className="mb-4">
          <button
            onClick={() => toggleSection(category.label)}
            className="flex w-full items-center justify-between px-2 py-1.5 text-[11px] tracking-wider text-text-0 transition-colors hover:text-accent"
            style={{ fontFamily: "var(--font-mono)", fontWeight: 700 }}
          >
            {toSnakeLabel(category.label)}
            <svg
              className={`h-3 w-3 transform text-text-3 transition-transform duration-200 ${
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
          <div
            className="overflow-hidden transition-all duration-200"
            style={{
              maxHeight: collapsed[category.label] ? "0px" : "1000px",
              opacity: collapsed[category.label] ? 0 : 1,
            }}
          >
            <div className="mt-1">
              {category.subcategories.map((sub) => (
                <div key={sub.label} className="mb-1">
                  <span
                    className="block px-2 py-1 text-[10px] tracking-wider text-text-3"
                    style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}
                  >
                    {toSnakeLabel(sub.label)}
                  </span>
                  <ul>
                    {sub.entries.map((entry) => {
                      const isActive = activeId === entry.id;
                      return (
                        <li key={entry.id}>
                          <button
                            onClick={() => handleClick(entry.id)}
                            className={`block w-full truncate rounded-md px-3 py-1 text-left text-[13px] transition-colors ${
                              isActive
                                ? "sidebar-active-link font-medium"
                                : "text-text-2 hover:bg-bg-3/50 hover:text-text-1"
                            }`}
                            style={{ fontFamily: "var(--font-mono)" }}
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
          </div>
        </div>
      ))}
    </nav>
  );

  return (
    <>
      {/* Mobile toggle */}
      <button
        onClick={() => setMobileOpen(!mobileOpen)}
        className="fixed bottom-4 right-4 z-50 flex h-12 w-12 items-center justify-center rounded-full shadow-lg lg:hidden"
        style={{
          background: "var(--color-accent)",
          color: "var(--color-bg-0)",
        }}
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
          className="search-overlay fixed inset-0 z-40 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`sidebar-hidden-scrollbar fixed top-16 z-40 h-[calc(100vh-4rem)] w-[248px] overflow-y-auto transition-transform lg:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
        style={{
          background: "var(--color-bg-1)",
          boxShadow: "1px 0 0 var(--color-border)",
        }}
      >
        {sidebarContent}
      </aside>
    </>
  );
}
