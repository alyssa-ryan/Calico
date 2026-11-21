import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { API_FINGERPRINT } from "../api/fingerprint";
import { renderAt, testClient } from "../test/render";
import { Header } from "./Header";

vi.mock("../api/client", async (original) => {
  const actual = await original<typeof import("../api/client")>();
  return { ...actual, api: { health: vi.fn() } };
});
const mocked = vi.mocked(api);

beforeEach(() => vi.resetAllMocks());

describe("Header", () => {
  it("asks for a restart when the server runs another version of Calico", async () => {
    mocked.health.mockResolvedValue({ engine: "ready", error: null, api: "0000000000000000" });
    renderAt(<Header />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("restart calico serve");
    expect(alert).toHaveTextContent("reload this page");
  });

  it("says nothing when the page and the server match", async () => {
    mocked.health.mockResolvedValue({ engine: "ready", error: null, api: API_FINGERPRINT });
    renderAt(<Header />);
    expect(await screen.findByText("Ready")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("says Disconnected when the server stops answering, though it answered before", async () => {
    const client = testClient();
    // An answer from long ago, so the header asks again, as it does every 30 seconds.
    client.setQueryData(
      ["health"],
      { engine: "ready", error: null, api: API_FINGERPRINT },
      { updatedAt: 0 },
    );
    mocked.health.mockRejectedValue(new TypeError("Failed to fetch"));
    renderAt(<Header />, { client });
    expect(await screen.findByText("Disconnected")).toBeInTheDocument();
    expect(screen.queryByText("Ready")).toBeNull();
  });
});
