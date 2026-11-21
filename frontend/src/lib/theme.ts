import { useCallback, useState } from "react";

export type Theme = "light" | "dark";
const KEY = "calico-theme";

const current = (): Theme =>
  document.documentElement.classList.contains("dark") ? "dark" : "light";

/** The page theme; the initial value was applied before paint by index.html. */
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(current);
  const toggle = useCallback(() => {
    const next: Theme = current() === "dark" ? "light" : "dark";
    document.documentElement.classList.toggle("dark", next === "dark");
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* private mode: the choice just isn't remembered */
    }
    setTheme(next);
  }, []);
  return [theme, toggle];
}
