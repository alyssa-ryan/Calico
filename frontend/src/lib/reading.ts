/** How "Describe it" reads a description, shown as the user types. */
import { useEffect, useState } from "react";
import type { ReadingPart, ReadingResponse } from "../api/types";

/** The parts worth showing: only when the reading found more than words to look for. */
export function readingToShow(reading: ReadingResponse | undefined): ReadingPart[] | null {
  if (!reading?.used) return null;
  const plain = reading.parts.length === 1 && reading.parts[0]?.kind === "looks for";
  return reading.parts.length === 0 || plain ? null : reading.parts;
}

/** ``value`` once it has stopped changing for ``ms`` milliseconds. */
export function useDebounced<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return settled;
}
