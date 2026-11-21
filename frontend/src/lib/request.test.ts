import { buildFormData, buildRequest, submitBlocker } from "./request";

const draft = {
  task: "triptych",
  folders: ["/a/x", "/a", "/b"],
  criteria: "  warm light ",
  onlyClearMatches: true,
  describeWeight: 0.4,
  template: "yours" as const,
  examples: ["/a/x/IMG_2.jpg"],
  limit: 6,
  options: { diversity: 0.5 },
  archiveReferences: ["/a/x/IMG_1.jpg"],
};

describe("request building", () => {
  it("sends top-most folders and trimmed criteria", () => {
    expect(buildRequest(draft)).toEqual({
      task: "triptych",
      folders: ["/a", "/b"],
      criteria: "warm light",
      only_clear_matches: true,
      describe_weight: 0.4,
      template: "yours",
      examples: ["/a/x/IMG_2.jpg"],
      limit: 6,
      options: { diversity: 0.5 },
      references: ["/a/x/IMG_1.jpg"],
    });
  });

  it("asks for clear matches only when there is a description to match", () => {
    expect(buildRequest({ ...draft, criteria: "  " }).only_clear_matches).toBe(false);
    expect(buildRequest({ ...draft, onlyClearMatches: false }).only_clear_matches).toBe(false);
  });

  it("says how much the description counts only when there is one", () => {
    expect(buildRequest({ ...draft, criteria: "  " }).describe_weight).toBe(0.4); // examples
    expect(buildRequest({ ...draft, criteria: "", examples: [] }).describe_weight).toBeNull();
    expect(buildRequest({ ...draft, describeWeight: null }).describe_weight).toBeNull();
  });

  it("builds multipart with a JSON request field and one field per upload", async () => {
    const files = [new File(["a"], "one.jpg"), new File(["b"], "two.jpg")];
    const form = buildFormData(buildRequest(draft), files);
    expect(JSON.parse(String(form.get("request"))).task).toBe("triptych");
    const refs = form.getAll("references") as File[];
    expect(refs.map((f) => f.name)).toEqual(["one.jpg", "two.jpg"]);
  });
});

describe("submitBlocker", () => {
  const ok = {
    folders: ["/a"],
    referenceUse: "optional" as const,
    referenceLabel: "Anchor photo",
    referenceCount: 0,
    optionErrors: {},
    limitValid: true,
  };
  it("is null when valid", () => expect(submitBlocker(ok)).toBeNull());
  it("needs a folder", () =>
    expect(submitBlocker({ ...ok, folders: [] })).toBe("Choose at least one folder."));
  it("needs a required reference, named by the task's label", () =>
    expect(submitBlocker({ ...ok, referenceUse: "required" })).toBe(
      "Add a anchor photo.".replace("a anchor", "an anchor"),
    ));
  it("surfaces option errors", () =>
    expect(submitBlocker({ ...ok, optionErrors: { c: "Pick at least one." } })).toBe(
      "Fine-tune: Pick at least one.",
    ));
});
