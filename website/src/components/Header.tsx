import Link from "next/link";
import { SearchModal } from "@/components/SearchModal";

function LogoBlocks() {
  return (
    <svg width="50" height="12" viewBox="0 0 100 24" xmlns="http://www.w3.org/2000/svg" className="block">
      <line x1="0" y1="12" x2="100" y2="12" stroke="#505058" strokeWidth="2" strokeLinecap="round" strokeDasharray="4 3"/>
      <rect x="8" y="2" width="20" height="20" rx="4.5" fill="#e8975d"/>
      <rect x="40" y="2" width="20" height="20" rx="4.5" fill="#7daa8c"/>
      <rect x="72" y="2" width="20" height="20" rx="4.5" fill="#8097c4"/>
    </svg>
  );
}

export function Header() {
  return (
    <header className="header-glass sticky top-0 z-50">
      <div className="mx-auto flex h-16 w-full max-w-[1440px] items-center justify-between px-4 sm:px-6">
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
