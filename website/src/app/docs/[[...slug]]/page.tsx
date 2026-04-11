import { notFound } from "next/navigation";
import Link from "next/link";
import { getDocPage, getAllDocSlugs, extractHeadings } from "@/lib/content";
import { getPrevNext } from "@/lib/navigation";
// MarkdocRenderer unused -- we render pre-highlighted HTML directly
import { TableOfContents } from "@/components/TableOfContents";

export async function generateStaticParams() {
  const slugs = getAllDocSlugs();
  return slugs.map((slug) => ({ slug }));
}

export default async function DocsPage({
  params,
}: {
  params: Promise<{ slug?: string[] }>;
}) {
  const { slug } = await params;
  const segments = slug ?? [];
  const page = getDocPage(segments);

  if (!page) {
    notFound();
  }

  const href =
    segments.length === 0 ? "/docs" : `/docs/${segments.join("/")}`;
  const { prev, next } = getPrevNext(href);
  const headings = extractHeadings(segments);

  return (
    <div className="mx-auto flex max-w-5xl gap-10 px-6 py-10 lg:px-10">
      {/* Article */}
      <article className="min-w-0 flex-1">
        <div
          className="prose"
          dangerouslySetInnerHTML={{ __html: page.htmlContent }}
        />

        {/* Prev / Next navigation */}
        <div
          className="mt-12 flex items-center justify-between pt-6 text-sm"
          style={{ borderTop: "1px solid var(--color-border)" }}
        >
          {prev ? (
            <Link
              href={prev.href}
              className="group flex items-center gap-1 text-text-2 transition-colors hover:text-accent"
            >
              <span className="transition-transform group-hover:-translate-x-0.5">
                &larr;
              </span>
              {prev.title}
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link
              href={next.href}
              className="group flex items-center gap-1 text-text-2 transition-colors hover:text-accent"
            >
              {next.title}
              <span className="transition-transform group-hover:translate-x-0.5">
                &rarr;
              </span>
            </Link>
          ) : (
            <span />
          )}
        </div>
      </article>

      {/* Right TOC */}
      <div className="hidden w-52 shrink-0 xl:block">
        <TableOfContents headings={headings} />
      </div>
    </div>
  );
}
