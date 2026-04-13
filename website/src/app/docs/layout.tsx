import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen flex-col" style={{ background: "var(--color-bg-0)" }}>
      <Header />
      <div className="flex flex-1">
        <Sidebar />
        {/* Main content area offset by sidebar width on large screens */}
        <main className="min-w-0 flex-1 overflow-y-auto lg:pl-[248px]">{children}</main>
      </div>
    </div>
  );
}
