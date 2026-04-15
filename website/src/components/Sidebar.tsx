"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { navigation } from "@/lib/navigation";

/** Convert "Getting Started" to "getting_started" for display */
function toSnakeCase(str: string): string {
  return str.toLowerCase().replace(/\s+/g, "_");
}

export function Sidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  // Track which sections are collapsed (all open by default)
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const toggleSection = (title: string) => {
    setCollapsed((prev) => ({ ...prev, [title]: !prev[title] }));
  };

  const sidebarContent = (
    <nav className="px-4 py-6">
      {navigation.map((section) => (
        <div key={section.title} className="mb-5">
          <button
            onClick={() => toggleSection(section.title)}
            className="flex w-full items-center justify-between px-1 py-1 text-[11px] tracking-wider text-text-2 transition-colors hover:text-text-1"
            style={{ fontFamily: "var(--font-mono)", fontWeight: 600, textTransform: "none" }}
          >
            {toSnakeCase(section.title)}
            <svg
              className={`h-3 w-3 transform transition-transform duration-200 ${
                collapsed[section.title] ? "-rotate-90" : ""
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
            className="grid transition-all duration-200"
            style={{
              gridTemplateRows: collapsed[section.title] ? "0fr" : "1fr",
              opacity: collapsed[section.title] ? 0 : 1,
            }}
          >
            <ul className="mt-1.5 min-h-0 overflow-hidden space-y-0.5">
              {section.items.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      onClick={() => setMobileOpen(false)}
                      className={`block rounded-md px-3 py-1.5 text-sm transition-colors ${
                        isActive
                          ? "sidebar-active-link font-medium"
                          : "text-text-2 hover:bg-bg-3/50 hover:text-text-1"
                      }`}
                    >
                      {item.title}
                    </Link>
                  </li>
                );
              })}
            </ul>
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
        aria-label="Toggle navigation"
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

      {/* Sidebar panel */}
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
