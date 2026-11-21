/** Labelling a search or rating stories: keys, progress and what to look at next. Pure. */
import type { Label, LabelPhoto, LabelSetDetail, LabelStory } from "../api/types";

export type LabelKind = LabelSetDetail["kind"];
export type LabelChoice = { value: Label; name: string; key: string };

const NAMES: Record<LabelKind, [string, string, string]> = {
  describe: ["Matches", "Partly", "No"],
  same_view: ["Same view", "Nearly", "No"],
  taste: ["Like", "Not sure", "Don't like"],
  stories: ["Works", "Not sure", "Doesn't"],
  compare: ["Like", "Not sure", "Don't like"],
  anchor: ["Could anchor", "Not sure", "Could not"],
};

/** The three choices, named for what is judged: a description, the same view, your taste
 * (alone or among first picks), a story, or whether a photo could anchor one. */
export function labelChoices(kind: LabelKind): LabelChoice[] {
  const [yes, partly, no] = NAMES[kind];
  return [
    { value: 2, name: yes, key: "1" },
    { value: 1, name: partly, key: "2" },
    { value: 0, name: no, key: "3" },
  ];
}

const KEYS: Record<string, Label> = { "1": 2, y: 2, "2": 1, p: 1, "3": 0, n: 0 };

/** 1/2/3 (or y/p/n) for the first, second and third choice. */
export const labelForKey = (key: string): Label | undefined => KEYS[key.toLowerCase()];

type Labelled = { label: Label | null };

/** The first unlabelled photo (or story) after ``from``, wrapping round; -1 when all are. */
export function nextUnlabelled(items: readonly Labelled[], from: number): number {
  for (let step = 1; step <= items.length; step++) {
    const i = (from + step) % items.length;
    if (items[i]?.label === null) return i;
  }
  return -1;
}

export function withLabel(
  items: readonly LabelPhoto[],
  contentKey: string,
  label: Label | null,
): LabelPhoto[] {
  return items.map((item) => (item.content_key === contentKey ? { ...item, label } : item));
}

export function labelProgress(items: readonly Labelled[]) {
  const labelled = items.filter((i) => i.label !== null).length;
  return {
    labelled,
    total: items.length,
    text: `${labelled} of ${items.length} labelled`,
    done: labelled === items.length,
  };
}

/** Content keys of a story's photos marked as not fitting it. */
export const storyMisfits = (story: LabelStory): string[] =>
  story.photos.filter((p) => !p.fits).map((p) => p.content_key);

/** The story's misfits once ``contentKey`` is marked, or unmarked if it was. */
export function toggledFit(story: LabelStory, contentKey: string): string[] {
  const now = storyMisfits(story);
  return now.includes(contentKey) ? now.filter((k) => k !== contentKey) : [...now, contentKey];
}

export function withStory(
  stories: readonly LabelStory[],
  index: number,
  label: Label | null,
  doesntFit: readonly string[],
): LabelStory[] {
  return stories.map((story) =>
    story.index === index
      ? {
          ...story,
          label,
          photos: story.photos.map((p) => ({ ...p, fits: !doesntFit.includes(p.content_key) })),
        }
      : story,
  );
}

/** Whether story ``index`` is rated ``label`` with exactly ``doesntFit`` not fitting: a
 * failed save is undone only while nothing newer has replaced it. */
export function storyIs(
  stories: readonly LabelStory[],
  index: number,
  label: Label | null,
  doesntFit: readonly string[],
): boolean {
  const story = stories.find((s) => s.index === index);
  if (!story) return false;
  const misfits = storyMisfits(story);
  return (
    story.label === label &&
    misfits.length === doesntFit.length &&
    misfits.every((k) => doesntFit.includes(k))
  );
}
