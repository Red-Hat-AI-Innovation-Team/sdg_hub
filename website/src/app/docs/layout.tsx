import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <div className="flex flex-1">
        <Sidebar />
        {/* Main content area offset by sidebar width on large screens */}
        <main className="flex-1 lg:pl-[260px]">{children}</main>
      </div>
    </div>
  );
}
