import type { Metadata } from "next";
import { Barlow, Source_Sans_3 } from "next/font/google";
import "./globals.css";

const barlow = Barlow({
  subsets: ["latin"],
  variable: "--font-barlow",
  weight: ["500", "600", "700", "800"],
});

const sourceSans = Source_Sans_3({
  subsets: ["latin"],
  variable: "--font-source-sans",
  weight: ["300", "400", "600"],
});

export const metadata: Metadata = {
  title: "IIT International Student Assistant",
  description: "Ask questions about F-1 visa, CPT, OPT, STEM OPT, travel, and more — Illinois Institute of Technology.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${barlow.variable} ${sourceSans.variable}`}>
      <body className="font-body antialiased">{children}</body>
    </html>
  );
}
