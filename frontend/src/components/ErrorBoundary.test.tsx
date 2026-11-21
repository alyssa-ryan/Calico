import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { APP_ERROR } from "../lib/format";
import { ErrorBoundary } from "./ErrorBoundary";

function Broken(): never {
  throw new Error("result.groups is undefined");
}

describe("ErrorBoundary", () => {
  it("shows what went wrong, and how to recover, instead of a blank page", () => {
    vi.spyOn(console, "error").mockImplementation(() => {}); // React logs the error too
    render(
      <ErrorBoundary>
        <Broken />
      </ErrorBoundary>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(APP_ERROR.title);
    expect(screen.getByRole("alert")).toHaveTextContent("result.groups is undefined");
    expect(screen.getByRole("button", { name: APP_ERROR.reload })).toBeInTheDocument();
  });
});
