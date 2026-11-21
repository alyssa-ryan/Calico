import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { pageTitle } from "../lib/format";
import { RouteFocus } from "./RouteFocus";

describe("RouteFocus", () => {
  it("names the page and moves focus to its heading when you navigate", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/"]}>
        <RouteFocus />
        <Routes>
          <Route path="/" element={<Link to="/runs">Past runs</Link>} />
          <Route path="/runs" element={<h1>Past runs</h1>} />
        </Routes>
      </MemoryRouter>,
    );
    expect(document.title).toBe(pageTitle("/"));
    await user.click(screen.getByRole("link", { name: "Past runs" }));
    expect(await screen.findByRole("heading", { name: "Past runs" })).toHaveFocus();
    expect(document.title).toBe(pageTitle("/runs"));
  });

  it("titles every page", () => {
    expect(pageTitle("/")).toBe("New selection · Calico");
    expect(pageTitle("/runs/20261028-1")).toBe("A run · Calico");
    expect(pageTitle("/label/abc")).toBe("Labelling · Calico");
    expect(pageTitle("/nowhere")).toBe("Calico");
  });
});
