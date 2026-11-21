import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Footer, Masthead } from "./Banners";

describe("Banners", () => {
  it("shows the header artwork as decoration, not as content", () => {
    render(<Masthead />);
    const art = screen.getByTestId("masthead").querySelector("img");
    expect(art).toHaveAttribute("alt", "");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("offers each artwork at two sizes so phones download the small one", () => {
    render(
      <>
        <Masthead />
        <Footer />
      </>,
    );
    for (const art of document.querySelectorAll("img")) {
      expect(art.getAttribute("srcset")?.split(",")).toHaveLength(2);
      expect(art).toHaveAttribute("width");
      expect(art).toHaveAttribute("height");
    }
  });

  it("makes the footer the page's contentinfo landmark", () => {
    render(<Footer />);
    expect(screen.getByRole("contentinfo").querySelector("img")).toHaveAttribute("alt", "");
  });
});
