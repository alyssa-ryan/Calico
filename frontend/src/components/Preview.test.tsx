import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PREVIEW } from "../lib/format";
import { Preview } from "./Preview";

describe("Preview", () => {
  it("says the preview is unavailable when the image cannot be loaded", () => {
    render(<Preview src="/thumbs/gone" alt="a.jpg" fade />);
    const image = screen.getByRole("img", { name: "a.jpg" });
    expect(image).toHaveClass("opacity-0");
    fireEvent.error(image);
    expect(screen.getByRole("img", { name: "a.jpg" })).toHaveTextContent(PREVIEW.unavailable);
  });

  it("fades in once loaded", () => {
    render(<Preview src="/thumbs/k" alt="a.jpg" fade />);
    const image = screen.getByRole("img", { name: "a.jpg" });
    fireEvent.load(image);
    expect(image).toHaveClass("opacity-100");
  });

  it("tries again for another photo", () => {
    const { rerender } = render(<Preview src="/thumbs/a" alt="a.jpg" />);
    fireEvent.error(screen.getByRole("img", { name: "a.jpg" }));
    rerender(<Preview src="/thumbs/b" alt="b.jpg" />);
    expect(screen.getByRole("img", { name: "b.jpg" }).tagName).toBe("IMG");
  });
});
