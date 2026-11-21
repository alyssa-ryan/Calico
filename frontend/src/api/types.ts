import type { components } from "./schema";

type S = components["schemas"];

export type TaskInfo = S["TaskInfo"];
export type ReferenceUse = TaskInfo["reference_use"];
export type Health = S["Health"];
export type TasteInfo = S["TasteInfo"];
export type Template = TasteInfo["default"];
export type RootFolder = S["RootFolder"];
export type ChildFolder = S["ChildFolder"];
export type FolderMatch = S["FolderMatch"];
export type JobStatus = S["JobStatus"];
export type RunListItem = S["RunListItem"];
export type RunDetail = S["RunDetail"];
export type SelectionResult = S["SelectionResult"];
export type SelectionRequest = S["SelectionRequest"];
export type Group = S["Group"];
export type Pick = S["Pick"];
export type Evidence = S["Evidence"];
export type StageStat = S["StageStat"];
export type LabelSetSummary = S["LabelSetSummary"];
export type LabelSetDetail = S["LabelSetDetail"];
export type LabelPhoto = S["LabelPhoto"];
export type LabelStory = S["LabelStory"];
export type Label = 0 | 1 | 2;
export type ReadingResponse = S["ReadingResponse"];
export type ReadingPart = S["ReadingPart"];
export type Rating = -1 | 1;

export type SelectBody = {
  task: string;
  folders: string[];
  criteria: string;
  only_clear_matches: boolean;
  describe_weight: number | null;
  template: Template | null;
  examples: string[];
  limit: number | null;
  options: Record<string, unknown>;
  references: string[];
};
