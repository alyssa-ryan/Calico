import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { renderAt, TASKS } from "../test/render";
import { Compose } from "./Compose";

vi.mock("../api/client", async (original) => {
  const actual = await original<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      tasks: vi.fn(),
      roots: vi.fn(),
      children: vi.fn(),
      select: vi.fn(),
      job: vi.fn(),
      health: vi.fn(),
      reading: vi.fn(),
      taste: vi.fn(),
    },
  };
});

const mocked = vi.mocked(api);

beforeEach(() => {
  vi.resetAllMocks();
  mocked.tasks.mockResolvedValue(TASKS);
  mocked.taste.mockResolvedValue({ default: "general", trained: false });
  mocked.roots.mockResolvedValue([{ name: "archive", path: "/archive", mounted: true }]);
  mocked.children.mockResolvedValue([]);
  mocked.reading.mockResolvedValue({ used: true, parts: [{ kind: "looks for", text: "x" }] });
});

describe("Compose", () => {
  it("offers every task as a card", async () => {
    renderAt(<Compose />);
    expect(await screen.findAllByTestId("task-card")).toHaveLength(3);
  });

  it("asks for a folder before anything else", async () => {
    renderAt(<Compose />);
    expect(await screen.findByText("Choose at least one folder.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /find photographs/i })).toBeDisabled();
  });

  it("shows the reference zone, labelled by the task, only when the task uses one", async () => {
    const user = userEvent.setup();
    renderAt(<Compose />);
    await screen.findAllByTestId("task-card");
    expect(screen.queryByTestId("reference-drop")).not.toBeInTheDocument();

    await user.click(screen.getByText("Same view over time"));
    expect(screen.getByRole("heading", { name: "Reference photo" })).toBeInTheDocument();
    expect(screen.getByText("required")).toBeInTheDocument();

    await user.click(screen.getByText("Three-photo stories"));
    expect(screen.getByRole("heading", { name: "Anchor photo" })).toBeInTheDocument();
    expect(screen.getByText("optional")).toBeInTheDocument();
  });

  it("sends the request as multipart JSON and polls the job", async () => {
    const user = userEvent.setup();
    mocked.select.mockResolvedValue({ job_id: "j1" });
    mocked.job.mockResolvedValue({
      status: "running",
      done: 1,
      total: 4,
      step: "reading",
      run_id: null,
      label_set_id: null,
      error: null,
    });
    renderAt(<Compose />);
    await user.click(await screen.findByRole("checkbox", { name: /archive/ }));
    await user.type(screen.getByLabelText(/what are you looking for/i), "warm light");
    await user.click(screen.getByRole("button", { name: /find photographs/i }));

    await waitFor(() => expect(mocked.select).toHaveBeenCalledTimes(1));
    const form = mocked.select.mock.calls[0]?.[0] as FormData;
    expect(JSON.parse(String(form.get("request")))).toEqual({
      task: "best",
      folders: ["/archive"],
      criteria: "warm light",
      only_clear_matches: false,
      describe_weight: null,
      template: null,
      examples: [],
      limit: null,
      options: {},
      references: [],
    });
    expect(await screen.findByText("Reading photos: 1 of 4")).toBeInTheDocument();
  });

  it("starts one search however fast the button is pressed twice", async () => {
    const user = userEvent.setup();
    let answer!: (job: { job_id: string }) => void;
    mocked.select.mockReturnValue(new Promise((res) => (answer = res)));
    renderAt(<Compose />);
    await user.click(await screen.findByRole("checkbox", { name: /archive/ }));
    const find = screen.getByRole("button", { name: /find photographs/i });
    await user.dblClick(find);
    expect(mocked.select).toHaveBeenCalledTimes(1);
    answer({ job_id: "j1" });
  });

  it("does not start a search when Enter confirms the folder filter", async () => {
    const user = userEvent.setup();
    mocked.select.mockResolvedValue({ job_id: "j1" });
    renderAt(<Compose />);
    await user.click(await screen.findByRole("checkbox", { name: /archive/ }));
    await user.type(screen.getByRole("searchbox", { name: "Filter folders" }), "mill{Enter}");
    expect(mocked.select).not.toHaveBeenCalled();
  });

  it("leaves checking the options to Calico, so a value is never refused in silence", async () => {
    const { container } = renderAt(<Compose />);
    await screen.findByRole("checkbox", { name: /archive/ });
    expect(container.querySelector("form")?.noValidate).toBe(true);
  });

  it("offers General or Yours where taste ranks, with Yours off until it is trained", async () => {
    const user = userEvent.setup();
    renderAt(<Compose />);
    const taste = await screen.findByRole("radiogroup", { name: "Whose taste" });
    expect(within(taste).getByRole("radio", { name: /General/ })).toBeChecked();
    expect(within(taste).getByRole("radio", { name: /Yours/ })).toBeDisabled();
    expect(within(taste).getByText(/calico train taste/)).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: /Same view over time/ }));
    expect(screen.queryByRole("radiogroup", { name: "Whose taste" })).toBeNull();
  });

  it("sends Yours when chosen, and names it in the summary", async () => {
    const user = userEvent.setup();
    mocked.taste.mockResolvedValue({ default: "general", trained: true });
    mocked.select.mockResolvedValue({ job_id: "j1" });
    mocked.job.mockResolvedValue({
      status: "running",
      done: 0,
      total: 0,
      step: "reading",
      run_id: null,
      label_set_id: null,
      error: null,
    });
    renderAt(<Compose />);
    await user.click(await screen.findByRole("checkbox", { name: /archive/ }));
    const yours = await screen.findByRole("radio", { name: /Yours/ });
    await user.click(yours);
    expect(screen.getByTestId("summary")).toHaveTextContent("your taste");
    await user.click(screen.getByRole("button", { name: /find photographs/i }));
    await waitFor(() => expect(mocked.select).toHaveBeenCalledTimes(1));
    const form = mocked.select.mock.calls[0]?.[0] as FormData;
    expect(JSON.parse(String(form.get("request"))).template).toBe("yours");
  });

  it("starts from your saved template", async () => {
    mocked.taste.mockResolvedValue({ default: "yours", trained: true });
    renderAt(<Compose />);
    expect(await screen.findByRole("radio", { name: /Yours/ })).toBeChecked();
  });

  it("offers only clear matches once there is a description, and sends it", async () => {
    const user = userEvent.setup();
    mocked.select.mockResolvedValue({ job_id: "j1" });
    mocked.job.mockResolvedValue({
      status: "running",
      done: 0,
      total: 0,
      step: "reading",
      run_id: null,
      label_set_id: null,
      error: null,
    });
    renderAt(<Compose />);
    await user.click(await screen.findByRole("checkbox", { name: /archive/ }));
    expect(screen.queryByRole("checkbox", { name: /only clear matches/i })).toBeNull();
    await user.type(screen.getByLabelText(/what are you looking for/i), "a staircase");
    const only = screen.getByRole("checkbox", { name: /only clear matches/i });
    expect(only).not.toBeChecked();
    await user.click(only);
    expect(screen.getByTestId("summary")).toHaveTextContent("only clear matches");
    await user.click(screen.getByRole("button", { name: /find photographs/i }));
    await waitFor(() => expect(mocked.select).toHaveBeenCalledTimes(1));
    const form = mocked.select.mock.calls[0]?.[0] as FormData;
    expect(JSON.parse(String(form.get("request"))).only_clear_matches).toBe(true);
  });

  it("sets how much the description counts beside the box, from the task's default", async () => {
    const user = userEvent.setup();
    mocked.select.mockResolvedValue({ job_id: "j1" });
    mocked.job.mockResolvedValue({
      status: "running",
      done: 0,
      total: 0,
      step: "reading",
      run_id: null,
      label_set_id: null,
      error: null,
    });
    renderAt(<Compose />);
    await user.click(await screen.findByRole("checkbox", { name: /archive/ }));
    const name = /how much the description counts/i;
    expect(screen.queryByRole("slider", { name })).toBeNull();
    await user.type(screen.getByLabelText(/what are you looking for/i), "a staircase");
    const slider = screen.getByRole("slider", { name });
    expect(slider).toHaveValue("0.5");
    expect(screen.getByTestId("describe-weight")).toHaveTextContent("50%");

    await user.click(screen.getByText("Same view over time"));
    expect(screen.getByRole("slider", { name })).toHaveValue("0.3");
    await user.click(screen.getByText("Strongest photographs"));

    fireEvent.change(screen.getByRole("slider", { name }), { target: { value: "0.8" } });
    expect(screen.getByTestId("describe-weight")).toHaveTextContent("80%");
    expect(screen.getByTestId("summary")).toHaveTextContent("counts 80%");
    await user.click(screen.getByRole("button", { name: "default (50%)" }));
    expect(screen.getByRole("slider", { name })).toHaveValue("0.5");

    fireEvent.change(screen.getByRole("slider", { name }), { target: { value: "0.8" } });
    await user.click(screen.getByRole("button", { name: /find photographs/i }));
    await waitFor(() => expect(mocked.select).toHaveBeenCalledTimes(1));
    const form = mocked.select.mock.calls[0]?.[0] as FormData;
    expect(JSON.parse(String(form.get("request"))).describe_weight).toBe(0.8);
  });

  it("shows examples from a result as part of the description, removable, and sends them", async () => {
    const user = userEvent.setup();
    mocked.select.mockResolvedValue({ job_id: "j1" });
    mocked.job.mockResolvedValue({
      status: "running",
      done: 0,
      total: 0,
      step: "reading",
      run_id: null,
      label_set_id: null,
      error: null,
    });
    const examples = [
      { path: "/archive/a.jpg", contentKey: "ka" },
      { path: "/archive/b.jpg", contentKey: "kb" },
    ];
    renderAt(<Compose />, { state: { task: "best", folders: ["/archive"], examples } });
    const list = await screen.findByRole("list", { name: "Like these photos" });
    expect(within(list).getAllByRole("img")).toHaveLength(2);
    expect(screen.getByRole("slider", { name: /how much the description counts/i })).toBeVisible();
    expect(screen.getByTestId("summary")).toHaveTextContent("like 2 examples");
    await user.click(screen.getByRole("button", { name: "Remove b.jpg" }));
    expect(within(list).getAllByRole("img")).toHaveLength(1);
    await user.click(screen.getByRole("button", { name: /find photographs/i }));
    await waitFor(() => expect(mocked.select).toHaveBeenCalledTimes(1));
    const form = mocked.select.mock.calls[0]?.[0] as FormData;
    expect(JSON.parse(String(form.get("request"))).examples).toEqual(["/archive/a.jpg"]);
  });

  it("shows how it reads a description once typing pauses", async () => {
    const user = userEvent.setup();
    mocked.reading.mockResolvedValue({
      used: true,
      parts: [
        { kind: "looks for", text: "a sky" },
        { kind: "without", text: "clouds" },
      ],
    });
    renderAt(<Compose />);
    await user.type(
      await screen.findByLabelText(/what are you looking for/i),
      "a sky without clouds",
    );
    const reading = await screen.findByTestId("reading");
    expect(reading).toHaveTextContent("without clouds");
    expect(mocked.reading).toHaveBeenLastCalledWith("a sky without clouds", []);
    expect(mocked.reading).toHaveBeenCalledTimes(1); // once typing paused, not per key
  });

  it('starts "more like this" with the search\'s own settings, and says what it left out', async () => {
    const user = userEvent.setup();
    mocked.select.mockResolvedValue({ job_id: "j1" });
    const state = {
      task: "best",
      folders: ["/archive"],
      criteria: "warm light",
      references: [],
      examples: [],
      onlyClearMatches: false,
      describeWeight: null,
      template: null,
      limit: 9,
      options: { diversity: 0.6 },
      dropped: ["references"],
    };
    renderAt(<Compose />, { state });
    expect(await screen.findByRole("note")).toHaveTextContent(/reference photo/i);
    await user.click(screen.getByRole("button", { name: /find photographs/i }));
    await waitFor(() => expect(mocked.select).toHaveBeenCalledTimes(1));
    const form = mocked.select.mock.calls[0]?.[0] as FormData;
    const sent = JSON.parse(String(form.get("request")));
    expect(sent).toMatchObject({ limit: 9, options: { diversity: 0.6 } });
  });

  it("starts pre-filled with an archive anchor from a result card", async () => {
    renderAt(<Compose />, {
      state: {
        task: "triptych",
        folders: ["/archive"],
        references: [{ path: "/archive/a.jpg", contentKey: "k" }],
      },
    });
    expect(await screen.findByRole("img", { name: "a.jpg" })).toBeInTheDocument();
    expect(screen.getByTestId("summary")).toHaveTextContent("anchor photo: a.jpg");
  });

  it("shows the server's reason when a job fails", async () => {
    const user = userEvent.setup();
    mocked.select.mockResolvedValue({ job_id: "j1" });
    mocked.job.mockResolvedValue({
      status: "error",
      done: 0,
      total: 0,
      step: "reading",
      run_id: null,
      label_set_id: null,
      error: "Folder is not mounted.",
    });
    renderAt(<Compose />);
    await user.click(await screen.findByRole("checkbox", { name: /archive/ }));
    await user.click(screen.getByRole("button", { name: /find photographs/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Folder is not mounted.");
  });
});
