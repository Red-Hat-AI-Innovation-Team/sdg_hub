import { Header } from "@/components/Header";

export default function ApiReferenceLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen flex-col" style={{ background: "var(--color-bg-0)" }}>
      <Header />
      {children}
    </div>
  );
}
