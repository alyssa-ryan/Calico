/** Display helpers. Pure functions: everything the UI shows as text goes through here. */
import type { Evidence, Pick } from "../api/types";

export const percent = (value: number): string => `${Math.round(value * 100)}%`;

const dateFormat = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
});
const monthFormat = new Intl.DateTimeFormat("en-GB", { month: "long", year: "numeric" });

/** Capture times are camera-local wall-clock times (no zone), shown as such. */
export function formatDate(iso: string | null | undefined): string {
  return iso ? dateFormat.format(new Date(iso)) : "Undated";
}

export function fileName(path: string): string {
  return path.split("/").filter(Boolean).pop() ?? path;
}

export const signalLabel = (signal: string): string => signal.replace(/_/g, " ");

/** A pick's alternates: other frames of its burst, or of its view from other occasions. */
export const moreFrames = (count: number): string =>
  `+${count} more frame${count === 1 ? "" : "s"}`;

/** The model is confident this photo matches the description (see calico.criteria). */
export const clearMatchBadge = (pick: Pick): string | null =>
  pick.clear_match ? "Clear match" : null;

export const ONLY_CLEAR_MATCHES = "only clear matches";

/** Yours's probability that you like a pick, read from what you said at places it never learned. */
export const likelyBadge = (probability: number | null | undefined): string | null =>
  probability == null ? null : `${percent(probability)} likely`;

export const LIKELY = {
  help: "How likely you are to like it, from what you said at places your taste had not learned.",
};

/** The line where Yours's list stops being good: below it a pick is less likely than `probability`. */
export const MARKER = {
  line: (probability: number) =>
    `Below this line a pick is less than ${percent(probability)} likely to be one you like`,
  allAbove: (probability: number) =>
    `Every pick here is at least ${percent(probability)} likely to be one you like`,
};

/** Labelling a same-view run's photos, so `calico eval same-view` can set min_similarity. */
export const CALIBRATE = {
  button: "Label for calibration",
  choosing: "Choosing photos to label…",
  help: "Mark about 60 photos as the same view or not; calico eval same-view --patches then measures how well same views are found.",
};

/** Whose taste ranks photos: Calico's general one, or one learned from your ratings. */
export const TEMPLATE = {
  legend: "Whose taste",
  general: "General",
  generalHelp: "Calico's general sense of a strong photograph.",
  yours: "Yours",
  yoursHelp: "Learned from your ratings.",
  untrained:
    "Not trained yet: rate photos with Rate for your taste (Label), then run calico train taste.",
  chip: "your taste",
};

/** Rating photos for your taste, to train and test the Yours template. */
export const TASTE = {
  heading: "Rate for your taste",
  help: "About 60 photos from the folders you choose: mostly ones General or Yours would show first, above all where they differ, and some at random from the whole place. Rate one place at a time; six places give the Yours template enough to learn from and be tested on.",
  start: "Start rating",
  choosing: "Choosing photos to rate…",
  judge: "Judge only whether you like the photograph: would you want Calico to pick it?",
  choices:
    "Like: you would want it picked. Not sure: you can't decide (not used for training). Don't like: you would rather it was left out.",
};

/** Rating General's and Yours's first picks at a place, blind: whose do you prefer? */
export const COMPARE = {
  heading: "Compare Strongest",
  help: "About 30 first picks each from General and from Yours at the folders you choose, merged and shown in no particular order, without saying whose. Rate one place at a time; five or six places, travel among them, show whose first picks you prefer, and teach Yours.",
  start: "Compare first picks",
  choosing: "Choosing each template's first picks…",
  judge:
    "Judge each photograph as a first pick: would you want it among the first photos Calico shows you?",
  choices:
    "Like: you would want it among them. Not sure: you can't decide (left out). Don't like: you would rather another photo took its place.",
};

/** Rating whether photos could anchor a story, to learn what makes an anchor. */
export const ANCHORS = {
  heading: "Could it anchor a story?",
  help: "About 60 photos from the folders you choose: wide views, details and photos with people or animals in turn, from Calico's strongest to its weakest. Rate one place at a time; five or six places show whether Calico can already tell an anchor, or needs to learn it.",
  start: "Choose photos to rate",
  choosing: "Choosing photos to rate…",
  judge:
    "Judge each photograph as the middle of a three-photo story: could it hold the eye and say what ties the story together?",
  choices:
    "Could anchor: you could build a story around it. Not sure: you can't decide (left out). Could not: it could sit beside an anchor, or not in a story at all.",
};

/** Rating stories told from a place, to test how they are told and later learn what connects one. */
export const STORIES = {
  heading: "Rate stories",
  help: "About 20 stories from the folders you choose, told in several ways and shown in no particular order. Rate one place at a time; five or six places show whether the stories work for you.",
  start: "Tell stories to rate",
  choosing: "Telling stories to rate…",
  judge:
    "Judge each story as a whole: do the three photographs belong together, as a story you would want Calico to tell?",
  choices:
    "Works: you would want it told. Not sure: you can't decide (left out). Doesn't: the three don't belong together.",
  doesntFit:
    "Doesn't fit, under a photo, marks one that alone does not belong with the other two, whatever you think of the story.",
  misfit: "Doesn't fit",
  story: (n: number): string => `story ${n}`,
  position: (n: number, total: number): string => `Story ${n} of ${total}`,
};

/** The health check failed: the page cannot reach calico serve, so nothing done here saves. */
export const DISCONNECTED = {
  label: "Disconnected",
  help: "This page cannot reach calico serve, so nothing you do here is saved. Start it again (uv run calico serve), then reload this page.",
};

/** A job finished, but left no run or label set to open. */
export const JOB_MADE_NOTHING =
  "The request finished but made nothing to open. The details are in the server log; try again.";

export const STALE_SERVER =
  "Calico has changed since its server started, so requests may fail. Please restart calico serve (Ctrl+C, then uv run calico serve) and reload this page.";

/** How much the description counted: its share of each photo's score. */
export const describeWeightChip = (weight: number): string => `counts ${percent(weight)}`;

/** Example photos: part of the description, found from a result's "More like this". */
export const EXAMPLES = {
  label: "Like these photos",
  help: "Part of the description: Calico looks for more like these. They are left out of the results.",
  moreLikeThis: "More like this",
};

/** Shown when a page fails while drawing. */
export const APP_ERROR = {
  title: "This page could not be shown",
  reload: "Reload the page",
};

export const SKIP_TO_CONTENT = "Skip to content";

const PAGE_TITLES: [RegExp, string][] = [
  [/^\/$/, "New selection"],
  [/^\/runs\/?$/, "Past runs"],
  [/^\/runs\/[^/]+/, "A run"],
  [/^\/label\/?$/, "Label searches"],
  [/^\/label\/[^/]+/, "Labelling"],
];

/** The browser tab's title for the page at ``pathname``. */
export function pageTitle(pathname: string): string {
  const found = PAGE_TITLES.find(([pattern]) => pattern.test(pathname));
  return found ? `${found[1]} · Calico` : "Calico";
}

/** Shown in place of a photo whose image cannot be loaded. */
export const PREVIEW = { unavailable: "Preview unavailable" };

/** Said when a search started from a result could not bring all of the original along. */
export const PREFILL = {
  dropped: (parts: readonly string[]): string =>
    `This search starts from the one you came from, except its ${parts
      .map((p) => DROPPED[p] ?? p)
      .join(" and ")}: add ${parts.length > 1 ? "them" : "it"} again if you want ${
      parts.length > 1 ? "them" : "it"
    }.`,
};

const DROPPED: Record<string, string> = { references: "reference photos" };

export const examplesChip = (count: number): string =>
  `like ${count} example${count === 1 ? "" : "s"}`;

export const DESCRIBE_WEIGHT = {
  label: "How much the description counts",
  help: "Its share of each photo’s score. At 0% it only marks clear matches; at 100% nothing else counts.",
};

export type Chip = { label: string; value: string; detail: string };

/** A story's connections (everything except its photos' strength), strongest first. */
export function connectionChips(evidence: readonly Evidence[]): Chip[] {
  return evidence
    .filter((e) => e.signal !== "strength")
    .sort((a, b) => b.value - a.value)
    .map((e) => ({ label: signalLabel(e.signal), value: percent(e.value), detail: e.detail }));
}

export type MonthGroup = { label: string; picks: Pick[] };

/** Consecutive picks grouped by capture month; the task already ordered them by date. */
export function groupByMonth(picks: readonly Pick[]): MonthGroup[] {
  const groups: MonthGroup[] = [];
  for (const pick of picks) {
    const label = pick.captured_at ? monthFormat.format(new Date(pick.captured_at)) : "Undated";
    const last = groups[groups.length - 1];
    if (last && last.label === label) last.picks.push(pick);
    else groups.push({ label, picks: [pick] });
  }
  return groups;
}

/** Searching folder names across every collection, and ticking all the matches. */
export const FOLDER_SEARCH = {
  placeholder: "Search folders, e.g. State Hospital",
  searching: "Searching folders…",
  count: (n: number, query: string): string => `${n} folder${n === 1 ? "" : "s"} match “${query}”`,
  none: (query: string): string => `No folders match “${query}”.`,
  selectAll: (n: number): string => `Select all ${n}`,
  untickAll: (n: number): string => `Untick all ${n}`,
};

export const NO_MATCHES = "No photographs matched. The notes above say why.";

/** Same view's second tier: the same place, perhaps re-aimed, closest first. */
export const POSSIBLY_SAME = {
  hint: "The same place, perhaps re-aimed or from a little further off. Look them over: the closest come first.",
};

const photos = (n: number) => `${n} photo${n === 1 ? "" : "s"}`;

/** Same view's two tiers, named on the page so neither is missed. */
export const SAME_VIEW = {
  sureHeading: (n: number): string => `Same view · ${n}`,
  possibleHeading: (title: string, n: number): string => `${title} · ${n}`,
  summary: (sure: number, possible: number): string => {
    const first =
      sure === 0
        ? "No photo is surely the same view."
        : `${photos(sure)} ${sure === 1 ? "is" : "are"} the same view, oldest first.`;
    if (possible === 0) return first;
    const verb = possible === 1 ? "is" : "are";
    return sure === 0
      ? `${first} ${possible} ${verb} possibly the same view.`
      : `${first} ${possible} more ${verb} possibly the same view.`;
  },
  jump: "Jump to possibly the same view",
};
