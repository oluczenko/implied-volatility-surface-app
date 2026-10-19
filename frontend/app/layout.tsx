import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Volsurface — Implied Volatility Explorer",
  description: "Explore option volatility across strikes and maturities with interactive surfaces, volatility smiles and model-aware Greeks.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
