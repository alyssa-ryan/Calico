import { useEffect, useRef } from "react";
import { useLocation } from "react-router-dom";
import { pageTitle } from "../lib/format";

/**
 * On every navigation: title the page, start at its top, and move keyboard and screen
 * reader focus to its heading, as a page load would (a single-page app otherwise leaves
 * focus on the link that was clicked). Not on the first page: focus starts where it is.
 */
export function RouteFocus() {
  const { pathname } = useLocation();
  const first = useRef(true);
  useEffect(() => {
    document.title = pageTitle(pathname);
    if (first.current) {
      first.current = false;
      return;
    }
    window.scrollTo?.(0, 0);
    // The page may still be loading its heading; look for it once it has drawn.
    let tries = 0;
    const focus = () => {
      const heading = document.querySelector<HTMLElement>("main h1, h1");
      if (heading) {
        heading.tabIndex = -1;
        heading.focus();
      } else if (tries++ < 20) {
        timer = window.setTimeout(focus, 50);
      }
    };
    let timer = window.setTimeout(focus, 0);
    return () => window.clearTimeout(timer);
  }, [pathname]);
  return null;
}
