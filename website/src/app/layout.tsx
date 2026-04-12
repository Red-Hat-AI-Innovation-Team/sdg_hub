import type { Metadata, Viewport } from "next";
import { Fraunces, DM_Sans, Fragment_Mono } from "next/font/google";
import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
  axes: ["opsz"],
});

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
  display: "swap",
});

const fragmentMono = Fragment_Mono({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-fragment-mono",
  display: "swap",
});

const siteUrl = process.env.GITHUB_PAGES === "true"
  ? "https://red-hat-ai-innovation-team.github.io/sdg_hub"
  : "";

export const metadata: Metadata = {
  title: "SDG Hub",
  description:
    "A modular Python framework for building synthetic data generation pipelines using composable blocks and flows.",
  icons: {
    icon: "/favicon.svg",
  },
  openGraph: {
    title: "SDG Hub",
    description: "Composable blocks and flows for synthetic data generation",
    type: "website",
    images: [{ url: `${siteUrl}/social-card.png`, width: 1200, height: 630 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "SDG Hub",
    description: "Composable blocks and flows for synthetic data generation",
    images: [`${siteUrl}/social-card.png`],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${fraunces.variable} ${dmSans.variable} ${fragmentMono.variable} h-full antialiased`}
    >
      <body className="min-h-full w-full overflow-x-hidden flex flex-col">{children}</body>
    </html>
  );
}
