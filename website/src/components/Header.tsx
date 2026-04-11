import Link from "next/link";

export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-bg/95 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-[1440px] items-center justify-between px-6">
        {/* Logo */}
        <Link
          href="/"
          className="font-serif text-xl font-bold tracking-tight text-text"
        >
          SDG Hub
        </Link>

        {/* Navigation */}
        <nav className="flex items-center gap-6 text-sm font-medium">
          <Link
            href="/docs"
            className="text-text-muted transition-colors hover:text-text"
          >
            Docs
          </Link>
          <Link
            href="/api-reference"
            className="text-text-muted transition-colors hover:text-text"
          >
            API Reference
          </Link>
          <a
            href="https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub"
            target="_blank"
            rel="noopener noreferrer"
            className="text-text-muted transition-colors hover:text-text"
          >
            GitHub
            <span className="ml-1 inline-block translate-y-[-1px] text-[10px] opacity-50">
              &#8599;
            </span>
          </a>
        </nav>
      </div>
    </header>
  );
}
