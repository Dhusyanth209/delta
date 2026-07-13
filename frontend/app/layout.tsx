import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DELTA — Project Risk Prediction",
  description: "AI-powered project cost-overrun and delivery-risk prediction for IT services. Built with XGBoost, SHAP explainability, and real-time predictions.",
  keywords: "project management, risk prediction, cost overrun, XGBoost, SHAP, IT services",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
