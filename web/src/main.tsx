import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "./theme.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { initToken } from "./api";
import { initScheme } from "./scheme";

initToken();
initScheme();
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
