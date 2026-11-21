import {
  changedOptions,
  defaultValues,
  numberFromInput,
  parseOptionsSchema,
  validateOptions,
} from "./options";

const schema = {
  $defs: { Connection: { enum: ["theme", "place_type"], title: "Connection", type: "string" } },
  properties: {
    kind: {
      anyOf: [{ type: "string" }, { type: "null" }],
      default: null,
      description: "Only this kind",
      title: "Kind",
    },
    diversity: { default: 0.3, maximum: 1, minimum: 0, title: "Diversity", type: "number" },
    criteria_weight: { default: 1, maximum: 5, minimum: 0, title: "Criteria", type: "number" },
    candidates: { default: 150, maximum: 2000, minimum: 3, title: "Candidates", type: "integer" },
    collapse: { default: false, title: "Collapse", type: "boolean" },
    connections: {
      default: ["theme"],
      items: { $ref: "#/$defs/Connection" },
      minItems: 1,
      title: "Connections",
      type: "array",
    },
    mode: { enum: ["a", "b"], default: "a", type: "string" },
  },
  type: "object",
};

describe("options schema", () => {
  const fields = parseOptionsSchema(schema);

  it("keeps declaration order and maps kinds", () => {
    expect(fields.map((f) => [f.key, f.kind])).toEqual([
      ["kind", "text"],
      ["diversity", "range"],
      ["criteria_weight", "number"],
      ["candidates", "number"],
      ["collapse", "boolean"],
      ["connections", "multi"],
      ["mode", "enum"],
    ]);
  });

  it("resolves $ref items into choices with labels", () => {
    const multi = fields.find((f) => f.key === "connections");
    expect(multi?.kind === "multi" && multi.choices).toEqual([
      { value: "theme", label: "Theme" },
      { value: "place_type", label: "Place type" },
    ]);
    expect(multi?.kind === "multi" && multi.minItems).toBe(1);
  });

  it("humanizes missing titles and carries descriptions", () => {
    expect(fields[0]?.description).toBe("Only this kind");
    expect(fields.find((f) => f.key === "mode")?.title).toBe("Mode");
  });

  it("sends nothing when everything is default", () => {
    expect(changedOptions(fields, defaultValues(fields))).toEqual({});
  });

  it("sends only changed values; empty text is null", () => {
    const values = {
      ...defaultValues(fields),
      diversity: 0.5,
      kind: "  portrait ",
      connections: ["theme"],
    };
    expect(changedOptions(fields, values)).toEqual({ diversity: 0.5, kind: "portrait" });
    expect(changedOptions(fields, { ...values, connections: ["place_type", "theme"] })).toEqual({
      diversity: 0.5,
      kind: "portrait",
      connections: ["place_type", "theme"],
    });
    expect(changedOptions(fields, { ...values, kind: "  " })).toEqual({ diversity: 0.5 });
  });

  it("treats a cleared default-text field as unchanged", () => {
    expect(changedOptions(fields, { ...defaultValues(fields), kind: "" })).toEqual({});
  });

  it("enforces minItems and numeric bounds", () => {
    const values = { ...defaultValues(fields), connections: [], candidates: 1 };
    expect(validateOptions(fields, values)).toEqual({
      connections: "Pick at least one.",
      candidates: "At least 3.",
    });
    expect(validateOptions(fields, { ...values, candidates: Number.NaN }).candidates).toBe(
      "Enter a number.",
    );
    expect(validateOptions(fields, defaultValues(fields))).toEqual({});
  });

  it("lets a decimal be typed wherever the bounds allow it", () => {
    const gap = parseOptionsSchema({
      properties: {
        occasion_gap_hours: { default: 2, exclusiveMinimum: 0, maximum: 720, type: "number" },
      },
      type: "object",
    });
    const [field] = gap;
    expect(field).toMatchObject({ kind: "number", step: "any" });
    expect(validateOptions(gap, { occasion_gap_hours: 1.5 })).toEqual({});
    expect(validateOptions(gap, { occasion_gap_hours: 0 })).toEqual({
      occasion_gap_hours: "More than 0.",
    });
  });

  describe("a number the task lets you leave out", () => {
    const optional = parseOptionsSchema({
      properties: {
        anchors_tried: {
          anyOf: [{ maximum: 2000, minimum: 1, type: "integer" }, { type: "null" }],
          default: null,
        },
        candidates: { default: 1000, maximum: 2000, minimum: 3, type: "integer" },
      },
      type: "object",
    });
    const [left, required] = optional;

    it("is valid empty and sends nothing, so the task's own default applies", () => {
      expect(left).toMatchObject({ kind: "number", optional: true, default: null });
      expect(validateOptions(optional, defaultValues(optional))).toEqual({});
      expect(changedOptions(optional, defaultValues(optional))).toEqual({});
      expect(validateOptions(optional, { anchors_tried: 0, candidates: 1000 })).toEqual({
        anchors_tried: "At least 1.",
      });
    });

    it("clears to null, where a required number clears to not-a-number", () => {
      if (left?.kind !== "number" || required?.kind !== "number") throw new Error("numbers");
      expect(numberFromInput(left, "", Number.NaN)).toBeNull();
      expect(numberFromInput(left, "36", 36)).toBe(36);
      expect(numberFromInput(required, "", Number.NaN)).toBeNaN();
      expect(validateOptions(optional, { anchors_tried: null, candidates: Number.NaN })).toEqual({
        candidates: "Enter a number.",
      });
    });
  });
});
