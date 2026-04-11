"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { navigation } from "@/lib/navigation";

const allItems = navigation.flatMap((section) =>
  section.items.map((item) => ({
    ...item,
    section: section.title,
  }))
);

export function SearchModal() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const router = useRouter();

  const filtered = query.trim()
    ? allItems.filter(
        (item) =>
          item.title.toLowerCase().includes(query.toLowerCase()) ||
          item.section.toLowerCase().includes(query.toLowerCase())
      )
    : allItems;

  const handleOpen = useCallback(() => {
    setOpen(true);
    setQuery("");
  }, []);

  const handleClose = useCallback(() => {
    setOpen(false);
    setQuery("");
  }, []);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
      if (e.key === "Escape") {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <>
      {/* Search trigger button */}
      <button
        onClick={handleOpen}
        className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm text-text-2 transition-colors hover:text-text-1"
        style={{
          background: "var(--color-bg-3)",
          boxShadow: "0 0 0 1px var(--color-border)",
        }}
      >
        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <span className="hidden sm:inline">Search</span>
        <kbd
          className="ml-1 hidden rounded px-1.5 py-0.5 text-[10px] font-medium sm:inline-block"
          style={{
            background: "var(--color-bg-4)",
            color: "var(--color-text-3)",
            boxShadow: "0 0 0 1px var(--color-border-strong)",
          }}
        >
          /
        </kbd>
      </button>

      {/* Modal overlay */}
      {open && (
        <div className="search-overlay fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]" onClick={handleClose}>
          <div
            className="w-full max-w-lg rounded-xl p-0 shadow-2xl"
            style={{
              background: "var(--color-bg-2)",
              boxShadow: "0 0 0 1px var(--color-border-strong), 0 24px 48px rgba(0,0,0,0.5)",
            }}
            role="dialog"
            aria-modal="true"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Search input */}
            <div className="flex items-center gap-3 border-b px-4 py-3" style={{ borderColor: "var(--color-border)" }}>
              <svg className="h-5 w-5 shrink-0 text-text-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                autoFocus
                type="text"
                placeholder="Search documentation..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="flex-1 bg-transparent text-sm text-text-0 outline-none placeholder:text-text-3"
              />
              <kbd
                className="rounded px-1.5 py-0.5 text-[10px] font-medium"
                style={{
                  background: "var(--color-bg-4)",
                  color: "var(--color-text-3)",
                  boxShadow: "0 0 0 1px var(--color-border-strong)",
                }}
              >
                ESC
              </kbd>
            </div>

            {/* Results */}
            <div className="max-h-[300px] overflow-y-auto p-2">
              {filtered.length === 0 ? (
                <div className="px-4 py-8 text-center text-sm text-text-3">
                  No results found
                </div>
              ) : (
                filtered.map((item) => (
                  <button
                    key={item.href}
                    onClick={() => {
                      router.push(item.href);
                      handleClose();
                    }}
                    className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-bg-3"
                  >
                    <span className="text-text-2" style={{ fontFamily: "var(--font-mono)", fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      {item.section}
                    </span>
                    <span className="text-text-1">{item.title}</span>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
