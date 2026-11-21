import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { FolderPicker } from "./FolderPicker";

vi.mock("../api/client", async (original) => {
  const actual = await original<typeof import("../api/client")>();
  return { ...actual, api: { children: vi.fn(), searchFolders: vi.fn() } };
});

const mocked = vi.mocked(api);
const ROOTS = [{ name: "explore", path: "/E", mounted: true }];
const MATCHES = [
  { name: "Ashby State Hospital - June 2019", path: "/E/Ashby State Hospital - June 2019" },
  { name: "Linden State Hospital", path: "/E/Linden State Hospital" },
];

function Picker({ initial = [] as string[] }) {
  const [ticked, setTicked] = useState<string[]>(initial);
  return (
    <>
      <FolderPicker roots={ROOTS} ticked={ticked} onChange={setTicked} />
      <output data-testid="ticked">{ticked.join("|")}</output>
    </>
  );
}

function renderPicker(initial: string[] = []) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Picker initial={initial} />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.resetAllMocks();
  mocked.children.mockResolvedValue([]);
  mocked.searchFolders.mockResolvedValue(MATCHES);
});

describe("FolderPicker search", () => {
  it("searches every collection and ticks all matches in one click", async () => {
    const user = userEvent.setup();
    renderPicker();
    await user.type(screen.getByRole("searchbox"), "state hospital");

    expect(await screen.findByText("2 folders match “state hospital”")).toBeInTheDocument();
    expect(mocked.searchFolders).toHaveBeenLastCalledWith("state hospital");
    await user.click(screen.getByRole("button", { name: "Select all 2" }));

    expect(screen.getByTestId("ticked")).toHaveTextContent(MATCHES.map((m) => m.path).join("|"));
    await user.click(screen.getByRole("button", { name: "Untick all 2" }));
    expect(screen.getByTestId("ticked")).toBeEmptyDOMElement();
  });

  it("ticks one match on its own", async () => {
    const user = userEvent.setup();
    renderPicker();
    await user.type(screen.getByRole("searchbox"), "state");
    await user.click(await screen.findByRole("checkbox", { name: /Linden State Hospital/ }));
    expect(screen.getByTestId("ticked")).toHaveTextContent("/E/Linden State Hospital");
  });

  it("says when nothing matches, and offers nothing to select", async () => {
    mocked.searchFolders.mockResolvedValue([]);
    const user = userEvent.setup();
    renderPicker();
    await user.type(screen.getByRole("searchbox"), "lighthouse");
    expect(await screen.findByText("No folders match “lighthouse”.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Select all/ })).not.toBeInTheDocument();
  });

  it("shows the folder tree again when the search is cleared", async () => {
    const user = userEvent.setup();
    renderPicker();
    await user.type(screen.getByRole("searchbox"), "state");
    await screen.findByText(/folders match/);
    await user.clear(screen.getByRole("searchbox"));
    expect(screen.getByText("explore")).toBeInTheDocument();
    expect(screen.queryByText(/folders match/)).not.toBeInTheDocument();
  });
});
