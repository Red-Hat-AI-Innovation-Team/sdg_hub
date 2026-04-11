import Link from "next/link";
import { SearchModal } from "@/components/SearchModal";

function LogoBlocks() {
  return (
    <div className="flex gap-[3px]">
      <div className="h-[10px] w-[10px] rounded-[2px]" style={{ background: "#e8975d" }} />
      <div className="h-[10px] w-[10px] rounded-[2px]" style={{ background: "#7daa8c" }} />
      <div className="h-[10px] w-[10px] rounded-[2px]" style={{ background: "#8097c4" }} />
      <div className="h-[10px] w-[10px] rounded-[2px]" style={{ background: "#a88bb8" }} />
    </div>
  );
}

export function Header() {
  return (
    <header className="header-glass sticky top-0 z-50">
      <div className="mx-auto flex h-16 max-w-[1440px] items-center justify-between px-6">
        {/* Logo */}
        <Link
          href="/"
          className="flex items-center gap-2.5 text-text-0 transition-opacity hover:opacity-80"
        >
          <LogoBlocks />
          <span
            className="text-lg font-semibold tracking-tight"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            sdg hub
          </span>
        </Link>

        {/* Navigation */}
        <nav className="flex items-center gap-5 text-sm font-medium">
          <Link
            href="/docs"
            className="text-text-2 transition-colors hover:text-text-0"
          >
            Docs
          </Link>
          <Link
            href="/api-reference"
            className="text-text-2 transition-colors hover:text-text-0"
          >
            API Reference
          </Link>
          <a
            href="https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub"
            target="_blank"
            rel="noopener noreferrer"
            className="text-text-2 transition-colors hover:text-text-0"
          >
            GitHub
            <span className="ml-1 inline-block translate-y-[-1px] text-[10px] opacity-50">
              &#8599;
            </span>
          </a>
          <SearchModal />
        </nav>
      </div>
    </header>
  );
}
