# Architecture

Calico answers one kind of question: *given these folders and this request,
which photographs?* Everything in the codebase exists to answer that question
reliably, explain each answer, and make the next kind of question cheap to add.

```
                 ┌──────────────── interfaces ────────────────┐
                 │  cli.py (Typer)   web/ (FastAPI JSON API)  │
                 │  frontend/ (React + Tailwind SPA)          │
                 └───────────────┬────────────────────────────┘
                                 │ SelectionRequest
                         ┌───────▼────────┐
                         │   engine.py    │  resolve pool → index → features → task → save run
                         │  (sessions.py) │  rating sessions and `calico eval`, built on its steps
                         └─┬──────┬─────┬─┘
             ┌─────────────┘      │     └──────────────┐
     ┌───────▼───────┐    ┌───────▼───────┐    ┌───────▼───────┐
     │   catalog/    │    │   features/   │    │    tasks/     │  ← one file per menu entry
     │ scan, EXIF,   │    │ image → vector│    │ best,         │
     │ SQLite store, │    │ (versioned,   │    │ same_view,    │
     │ indexer       │    │  cached)      │    │ triptych      │
     └───────────────┘    └───────────────┘    └───────┬───────┘
                                                       │ uses
                                     ┌─────────────────┼─────────────────┐
                              ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
                              │  models/    │   │ selection/  │   │  runs.py    │
                              │ score/label │   │ bursts, MMR,│   │ every run   │
                              │ (zero-shot  │   │ scoring,    │   │ saved +     │
                              │  or trained)│   │ context     │   │ explainable │
                              └─────────────┘   └─────────────┘   └─────────────┘
```

## Vocabulary

What each domain word means and where it lives in the code, with the technical terms
beside them.

| Term | Meaning | Code |
|---|---|---|
| **Collection** | A folder tree of photos with a rule for naming locations | `settings.CollectionConfig` |
| **Location** | The place a photo belongs to, from its folder (optionally via regex) | `Photo.location` |
| **Pool** | The catalogued photos under the folders a request names | `TaskContext.photos` |
| **Extractor** | Image → fixed-length vector, versioned and cached by content | `features/` |
| **Embedder** | Extractor with unit-norm rows; may also encode text. CLIP is the main one; SigLIP reads "Describe it" | `features.base.Embedder`, `features/clip.py` |
| **Model** | Features → judgement: a score per photo, or label probabilities | `models/` |
| **Sub-model** | A hierarchical label, e.g. `establishing:interior` under `establishing` | `models.base.rollup` |
| **Taxonomy** | A TOML vocabulary of labels + text prompts = a zero-shot label model | `models/taxonomies/` |
| **Task** | One entry in the menu; turns a context into groups of picks | `tasks/` |
| **Burst** | Frames shot seconds apart of the same thing; selected as one | `selection.bursts` |
| **Occasion** | The photos of one view taken together: a new one starts when two in a row are more than `occasion_gap_hours` (2) apart, so a morning and an evening visit are two | `selection.view_groups` |
| **Copy** | One frame saved twice (a re-export, a HEIC also kept as JPEG): different files, the same pixels; counted once | `selection.copies` |
| **Evidence** | One weighted signal behind a pick; the score is their sum | `domain.Evidence` |
| **Run** | One executed request, saved with every per-photo signal | `runs.py` |
| **Criteria scorer** | "Describe it" text → a score per photo and, when the model can say, how likely each matches; swappable and versioned | `criteria.py` |
| **Clear match** | A photo the describe model is confident matches (SigLIP, p ≥ 0.5) | `criteria.CLEAR_MATCH`, `selection/clear_matches.py` |
| **Reading** | A description split into what to look for, avoid, when and where | `reading.py` |
| **Example** | A photo from your collections that is part of the description ("More like this"); left out of the results | `SelectionRequest.examples`, `criteria.blend_examples` |
| **Look** | A light, colour or mood word and the measurements it implies | `looks.py`, `looks.toml` |
| **Labelling set** | One search + a sample of photos you mark matches / partly / no; a reference photo + photos you mark same view / nearly / no; a taste session: photos you mark like / not sure / don't like; or a story session: stories you mark works / not sure / doesn't, and photos that don't fit | `labels.py` |
| **Review** | One question about one thing: `(use, scope, subject)`, e.g. *does this photo match "fog"?* or *👍/👎 on a General Strongest result*. Its newest answer is its rating; see "Ratings and reviews" | `reviews.py`, `ratings_store.py` |
| **Taste** (`template` in code) | Whose taste ranks photos: *General* (zero-shot aesthetic) or *Yours* (`taste`, learned from your ratings). Changes only the model that answers to "aesthetic" | `SelectionRequest.template`, `models.registry.WithTaste` |

## The request path

1. **Validate.** `SelectionRequest` is parsed by pydantic; the task parses its
   own `options` with its `options_model`. Unknown options are errors. The options
   come from two places: the **track** the search runs on (its task under its taste,
   `tracks.py`; `src/calico/tracks/<track>.toml` holds the options that track's
   champion ships with, decision 0019) and the request's own, which win. The track's
   version is recorded in the run (`versions["track:<track>"]`).
2. **Index.** Each folder must be inside a configured collection. The indexer
   brings the catalog up to date incrementally (unchanged files cost one
   `stat`), decoding each new image **once** and running every standard
   extractor on that decode, then writing a thumbnail. The standard extractors
   are technical, layout, palette and three embedders: `embedder` (CLIP, the
   main one), `describe_embedder` (SigLIP, reads "Describe it") and
   `view_embedder` (DINOv2, tells the same view from look-alikes for Same view).
3. **Gather.** Photos under the folders, de-duplicated by content key.
4. **Features.** One matrix per standard extractor, rows aligned with photos.
   Photos that failed to decode are dropped with a note.
5. **References and criteria.** Reference images get the same features (they
   need not be catalogued). The configured **criteria scorer**, reading with the
   `describe_embedder` (SigLIP), turns the "Describe it" text into one score per
   photo (`TaskContext.criteria_scores`), so tasks never see text. If the text
   cannot be read (no text encoder, too long) the scorer says so in a note —
   never silently ignored. **Examples** (photos chosen with "More like this")
   add `EXAMPLE_WEIGHT` (0.25) × each photo's mean SigLIP cosine to them; they
   are left out of the pool, and one from outside the chosen folders is noted.
6. **Clear matches.** When the model can say how likely each photo matches,
   `selection.clear_matches` notes when no photo clearly matches, and, if the
   request asks for *only clear matches*, narrows the pool to them (or keeps it
   whole, saying so, when there are none).
7. **Patches, if the task asks.** A task's `patch_shortlist` may name photos to
   compare with the references part by part (Same view: its likeliest matches).
   The engine reads their thumbnails, extracts the view embedder's patches
   (DINOv2: 1,024 a photo) and puts each photo's share of patches that are mutual
   best matches with a reference in `TaskContext.patch_scores`
   (`selection.patch_match`). About 0.1 s a photo, so only a shortlist. The
   engine remembers the shares for its last `PATCH_MEMORY` reference sets, so
   labelling a search for calibration, or re-running it, does not compare again;
   the job reports progress as `step: "comparing"`. A task's `patch_pairs` may
   instead name pairs of its own photos to compare with each other (recurring
   views); their shares go in `TaskContext.pair_shares` and are remembered in the
   catalog (`pair_shares`, by content and view-embedder version), so each pair is
   compared once, ever. Each photo's patches are read once per run into a scratch
   file in the Calico home (`patch_store.py`: 1.5 MB a photo, deleted after), and shares
   are saved every `PAIR_SAVE` pairs, so a stopped run keeps what it compared. The free
   space is checked first (`DiskSpaceError` says how much is needed), since the file is
   sparse and a full disk would otherwise crash the process mid-run.
8. **Task.** Pure function of `TaskContext` → `list[Group]`. Each pick is then
   marked with whether it clearly matches.
9. **Save.** `runs/<run_id>/result.json` + `signals.csv` (every pool photo,
   every recorded signal). The result goes back to the caller.

## Scoring conventions (all tasks)

* Every signal is in **[0, 1]**, higher is better. Raw values that have no
  absolute meaning (cosine to criteria text, sharpness) are converted with
  `selection.scoring.percentile_rank` *within the pool*.
* A pick's score is `Σ value × weight` over its evidence, with weights summing
  to 1. If a signal is unavailable (e.g. no criteria) its weight is
  redistributed and a note says so.
* **How much the description counts** is one setting for every task: the
  request's `describe_weight`, the criteria's share of each photo's score
  (of its *strength* in `triptych`, whose connections are scored apart). The
  task's other weights split the rest. Unset, it is the task's
  `default_describe_weight`; the engine records the number used in the run.
  At 0 the description only marks clear matches, and a note says so.
* **Bursts are collapsed before ranking diversity**: one representative per
  burst, the rest returned as `alternates` (in `best`, a view's frames from other
  occasions too).
* **Strength** (how good a photo is on its own) is computed in one place,
  `selection.strength.photo_strength`, used by `best` and `triptych`.
  (`same_view` ranks by similarity to the reference, not by strength.)
* **Diversity** (in `best`) is maximal marginal relevance over embeddings
  (`selection.diversity.mmr`), so the top 12 are not 12 views of one doorway.
* Ties break by path, so identical inputs give identical outputs.

## Tasks

### `best` — the strongest photographs
Ranks by a weighted mix of **technical quality** (`quality` model: sharpness,
exposure, clipping), **aesthetic** (`aesthetic` taxonomy, zero-shot), and
**criteria match** (cosine to the criteria text; half the score by
default). Optional `kind` restricts to
photos the `shot_kind` model labels as that kind (e.g. `portrait`,
`establishing:interior`). One result = one photo. Bursts collapse to their best
frame; MMR then spreads the result (`diversity`), choosing only among the
strongest `diversity_reach` times as many photos as asked for, so it never
brings in a weak photo for being unlike the rest. The photos it chooses are shown
strongest first (ties by path), not in the order it chose them, so the list reads
as a descending one.

**How likely you are to like each pick, and where the list stops being good.** Under Yours, a
pick carries a probability (`Pick.probability`): Yours's own p(you like it) read through its
calibration (`models.calibration`, below), which says how often you liked what it gave each
probability at places it never learned. It is a probability of the photo, not of its place in
the list, so it does not move with the pool (strength is a percentile within it). The marker
(`Group.marker`, option `marker_probability`, 0.8) is drawn above the first pick you are less
likely than that to like, and past the last when none is. General gives no probability: it
knows nothing of your ratings. Neither does a pick at a place Yours learned from, where it
reads too sure, nor any pick of a taste not yet calibrated; each says so in the run's notes.
A description reorders the list by what it matches, so with one there is a probability but
no line (0010).

**One pick per view** (`one_per_view`, on by default; owner's choice, 5 November 2026).
Strongest is a shortlist of portfolio-worthy candidates, as many as the request asks for,
so a view shot on two occasions takes one place in it. Among the strongest `VIEW_REACH` (2) times `diversity_reach`
times as many photos as asked for, those alike as a whole at one location (the view
embedder, `view_similarity` 0.82) are compared part by part (`patch_pairs`), and grouped
into views by average linkage of their shares (`view_share` 0.17), as recurring views
groups them, with the same constants (`selection.view_groups`). Each view's strongest
frame is the pick; its other frames, from any occasion, are its alternates. Twice the
reach because folded frames let weaker photos move up, and they must have been compared
too. Every pair is remembered in the catalog, so it is compared once ever, by Strongest or
by recurring views: the first search at a place costs about a minute (240 photos for a
first page of 40), later ones nothing. The taste sessions' draw and `eval taste`'s first
picks run Strongest in memory through the same step (`Engine.compare_parts`), so they
measure the Strongest a search shows. When the view embedder cannot compare patches,
views are not folded, and a note says so. `calico runs explain` names a folded frame "a
stronger frame of its view".

### `same_view` — the same place, over time
Needs a reference photo. The photos most alike as a whole (the view embedder,
DINOv2, cosine ≥ `shortlist_similarity` 0.75, at most `shortlist_size` 600) are
compared with it **part by part**: each photo's share of patches that are mutual
best matches with the reference's (`selection.patch_match`, computed by the engine
from thumbnails). Two tiers come back:

* **Same view** (share ≥ `sure_share`, 0.25): every such photo, bursts included
  unless `collapse_bursts`, **in chronological order**, so change over time reads
  left to right. One result = one photo (default cap 200).
* **Possibly the same view** (share ≥ `possible_share`, 0.18): the same place,
  perhaps re-aimed, closest first, at most `possible_limit` (60), to look over.

Two tiers because one line could not serve every site: look-alike rooms at one
place share as many patches as true repeats at another. On 19 labelled
references (155 same views), *same view* kept 73% of same views with 97% right
(65% of those from another visit), and the two tiers together held 154 of 155.
A re-aimed shot of the same spot is welcome, so it is not counted wrong. Notes say how many more were left out, or how close the
closest came. When the view embedder cannot compare patches, the whole-photo
match of before is used: **semantic** (embedding cosine) and **layout**
(gradient-orientation grid cosine) mixed by `layout_weight`, kept from
`min_similarity` (0.75).

### `recurring_views` — the views you keep coming back to
No reference. At each place, photos that look alike as a whole (the view embedder,
cosine ≥ `candidate_similarity` 0.82) and were taken more than an occasion apart
are compared with each other part by part (`patch_pairs`). They are grouped into
views by **average linkage** of their patch shares (`group_share` 0.17,
`selection.view_groups`): two groups join only if their photos share enough on
average, so a look-alike room cannot chain two views together. A view's photos are
split into **occasions** (`occasion_gap_hours`, 2), measured within the view, and
views shot on at least `min_occasions` (2) come back, most occasions first. One
result = one view: one photo per occasion, oldest first (the most typical of the
occasion, or the best match to the description), its other frames as alternates.
Undated photos are left out, and a copy of a frame counts once (a re-export carries
its export date, which would look like another occasion); an edit exported without camera
data (no camera recorded, sharing ≥ `EDIT_SHARE` 0.50 of its details with another
photo) takes its original's date, so it joins that occasion as another frame. Times are
compared on two clocks: the camera's, and UTC (`selection.clock`) where the camera wrote
its time zone or it can be borrowed (same camera in the folder, else the folder's one
zone). Photos close on either clock are one occasion, so a zone can join a visit but never
split one. Notes say how many of each. Groups are per location, so a place filed under several names needs
`location_aliases`.

Measured on four places: 64 of 69 labelled repeats grouped with their reference, 2
of 219 "not the same view" photos pulled in; reviewed by eye, 213 of 229 groups were
one view (often from more than one angle) and 8 more were similar views worth
seeing. The first run at a place costs about 0.24 s a photo and 4 ms a pair
(one large site: 32 minutes); the first run over a whole collection (19,991 photos, 692k
pairs) took 1 hour 41 minutes. Later runs reuse what was compared.

### `triptych` — three photographs that tell a story together
Chooses an anchor, then finds
the pair of companions maximizing *strength + connection*, where connection is
the mean pairwise similarity under the chosen **connections**: `theme`
(embedding), `location` (same location), `place_type`, `mood`, `surface`
(materials and how they have aged: peeling paint, brick, wallpaper, rust...),
`color` (palette), `kind_mix` (complementary shot kinds, e.g. establishing + detail +
portrait), `kind_same`. Members must be distinct (not the same burst, not
near-duplicates). Companions are reused across stories at most `max_uses` times. One result =
one story, anchor in the centre.

Connections reward alike photos, so a share of the connection side
(`variety_weight`) goes to **variety** (`selection.variety`): 1 while the three
photos' mean CLIP similarity is at most `too_alike`, falling to 0 at the
near-duplicate cut. It is judged on the set, since a liked story can hold one
close pair. It also orders an anchor's companion shortlist, which would
otherwise hold only the anchor's look-alikes. `calico eval stories` re-scores the
stories you rated, in a session (see "Rating stories") or with 👍 fits / 👎 doesn't
fit on a result, from the evidence saved with each (its sum is the score the task gave the
story, so `distinct` and roles count; variety is replaced in it,
0013): liked against rejected,
within each run or session. The two kinds of answer are reported apart, never pooled.
`calico eval finished-stories FILE` measures stories made outside Calico: each is
scored by the task's own formula (`TriptychTask.scorer`, over its working set
plus the story's photos) against every story made by putting one of the task's
candidates in one of its photos' places, and its photos are checked against the
candidates, since the task never uses a photo outside them.
`calico eval learned-stories [FILE]` asks whether the story's evidence would rank
better weighted by your answers than by hand (audit A10-14). Each story rated works or
doesn't in a session, and each finished story in `FILE` (as one that works), gets every
signal the task reads (`StoryScorer.signals`: strength, each connection, variety, each
role), over its own pool. A regularised logistic (`evaluation.learn_stories`,
`STORY_L2`) learns their weights. Each session is judged by weights learned at no place
it shares (`learned_apart`), against today's way, paired by session. With `--swaps`, a
third scorer also learns from each finished story with one photo swapped for another
candidate (`STORY_SWAPS_PER_SIDE` at each side), as stories that don't work. They are
never judged (owner's Q4). It measures only: the task still adds its evidence by hand.

**Distinct** (`distinct_weight`, 0.15 of the score; `docs/decisions/0004`) weighs against a story with two
photos of one moment: alike by the view embedder and shot within `distinct_within` (10
minutes) of each other. It is full while that pair is under `distinct_from` (0.80) alike,
falling to 0 at `distinct_until` (0.95). Your stories are three different images; the same
room from another angle belongs in a carousel. Two alike photos shot further apart are an
echo, which your finished stories use: of their 14 pairs at least 0.80 alike, 11 are more
than 10 minutes apart, 5 of them on another visit and 6 on the same one (5 November 2026;
a small sample). So time apart tells them from Calico's re-angled rooms, and whether it is
the same visit does not. It counts in both ways of telling, in the companions' shortlist, and as
its own evidence. Measure it
with `calico eval learned-stories --variant distinct_weight=…` (your rated stories,
re-scored, paired by session) and `calico eval finished-stories --variant …` (your own
stories against swaps).

**Anchors** are tried strongest first, the first `max(3 × limit, 10)` candidates. With
`anchor_by="likely"` they are tried by the anchor prior instead (`selection.anchors`:
half strength, half a person or animal or the widest view, as roles try them), a change
of anchors alone. `calico eval anchors` found the anchors you rated at 0.81 by the prior
and 0.68 by general strength (3 November 2026, 6 places), so sessions rate it
(`STORY_VARIANT`). `calico eval finished-stories` says, per story, whether its middle
photo is among the anchors tried ("anchors tried"): the swaps cannot judge anchors.
At 1,000 candidates it was 4 of 28 by strength and 7 by the prior (General; Yours 3 and
4, 5 November 2026): 18 anchors are too few for the better order to reach most of yours.
`anchors_tried` tries more (off by default), to be measured with `--variant
anchors_tried=…` and a timed search before sessions rate it. By strength, 36 reached 11
and 90 reached 17 of 28 (General; Yours 10 and 15); by the prior no more (5 November).

**With an anchor photo** (uploaded, or "Build a story around this" on any
result card) every story is built around it: the anchor is in the middle of
each, and the stories offer different companion pairs. Without one, the
strongest candidates take turns as anchors.

## How sure a measurement is

Every `calico eval` headline is printed as `value [lo, hi] (n places, n rated)`
(`evaluation/stats.py`). Ratings cluster by place: one place's photos look alike and are
rated in one sitting, so counting them as independent makes every number look surer than
it is. An interval resamples the places (or the searches, references, runs or finished
stories: the report names its unit), then the photos within each, 2,000 times with a fixed
seed (rule 7), and keeps the middle 95%. Each model's outputs are kept, not refitted.

Two rankers measured on the same photos are compared by their **paired difference**: both
are computed on every resample, so what they share (a hard place, an easy one) cancels.
Yours against General, each scorer against the first, each way of matching or telling
against the first or today's, each reading against text alone. A difference whose interval
holds zero is printed "could be zero" (or `?` in a table), and nothing is said to win or
beat. Pooled values (each place counting by its pairs or photos) are shown with the
**average of places** beside them, so a few big places cannot carry a number unseen.

Two exceptions are said where they apply: `eval examples` draws each search whole (its AUC
is already averaged over draws of examples), and `eval finished-stories` draws each story
whole (it is one answer). Fewer than five units (`stats.MIN_UNITS`) cannot show how far a
value could move, so such a range is printed as too few to trust.

**The hold-out.** A setting chosen on your ratings and reported on the same ratings looks
better than it is. `calico holdout draw` sets aside, once, by a seeded draw
(`evaluation/holdout.py`), 4 of the places judged for your taste, 5 same-view references and
2 story sessions (never more than a third of each), in `<home>/holdout.toml`, which is never
replaced. Every `calico eval` of taste, Compare, 👍/👎 ratings, anchors, same view and
stories then leaves them out and says so; `--holdout` reports on them alone, as a last
check of a final candidate. The saved taste is not learned from them either (owner's
choice, 29 October 2026), so a Compare session at a held-out place judges a taste that has
never seen it. Describe-it searches, examples and finished stories are not held out.

A hold-out is a list of names, but a photo is a file that can be filed under several
locations, and a trip is often filed under each of its sites and again under an umbrella folder
(10 November 2026: 62% of one held-out trip folder's files were also filed
under eight sites Yours learned from). So a photo belongs to every location its file is
filed under, not only the first path's: one also filed under a held-out location is that
location's. A location that shares `NEAR_SHARE` (0.1) or more of the smaller one's files, and
at least `NEAR_FILES` (3), with a held-out place is grouped with it, as a second name for the
place is, and left out with it (`Sessions._held_filing`, `evaluation.holdout.near_places`). It
is one hop, never a chain: a catch-all folder would join unrelated sites. A Compare session at
a location next to one the taste learned from is not fresh, and
`calico holdout draw --leave-out NAME` never draws a place, nor the places next to it, and
refuses a name the catalog has no location of, which would leave that place in the draw;
`--places N` sets how many are drawn and `--dry-run` shows the draw without writing it
(0009).

**Leaks.** A place is held out by its name, but its answers can reach training under
another one. A judged photo saved twice (a re-export, a HEIC also kept as JPEG) has a second
key, and with an embedding a copy with the same label is a near-perfect predictor; one trip
filed under two names is two places. So every photo that is a copy of a judged one
(`selection.copies`, on the view embedder's vectors) and every photo at a place
`calico locations` groups with the judged one (`catalog.place_names.similar_places`, among
the rated names) is never learned from while that place is judged, nor by the saved taste
when the place is held out (`evaluation.taste_apart`). `eval taste` counts both. In
`eval examples`, an example saved again (the same file in another folder, or a copy of its
frame) is never scored, in either protocol, and "shot within two minutes of one" is judged
on one clock (`selection.clock`), so a camera and a phone set to different zones compare.
Each row counts what was left out. An edit (cropped, recoloured) is not the same pixels, so
neither check catches it.

**What was tried.** Every `calico eval` adds a line to `<home>/experiments.jsonl` naming each
setting it measured (`calico/experiments.py`), and once more than one has been tried its
report says how many: the best of twelve settings tried on the same ratings looks better
than it is. Looks at the hold-out alone are counted apart, from the day the current one was
drawn: a hold-out retired and drawn again starts with none.

## Measuring "Describe it"

Changes to how descriptions are read are judged against your labels, not by eye.

1. **Label.** The web interface's *Label* page (or `calico eval new TEXT FOLDER`)
   samples ~60 photos for a search: the current top 36, 12 from just below,
   12 from the rest (`evaluation.sample_for_labelling`), shown shuffled. You
   mark each 2 matches / 1 partly / 0 no. The sample is saved as `labels/<set_id>.json`,
   your answers in the ratings journal.
2. **Evaluate.** `calico eval criteria` re-scores every labelled search with
   each registered scorer, read by each text model (`--embedder`; default: the
   describe embedder, then the others), and reports precision@10, nDCG@10
   (both: are the first results right?), AUC (does a match outscore a
   non-match anywhere?), the best separating threshold, and **clear matches**:
   of the photos the model called a clear match, how many you labelled Matches
   and how many No.
3. **Promote.** A new scorer is added to `criteria.SCORERS` under a new name,
   measured beside `baseline`, and becomes the default (`criteria_scorer` in
   `calico.toml`) only if it is better on your labels.

Scorers should score each photo on its own (not relative to the pool), since
evaluation scores only the labelled sample.

**Examples.** `calico eval examples` asks whether example photos would help:
each search is read by its text, by text plus examples
(`criteria.blend_examples`: the text's score plus a weight × the photo's mean
cosine to the examples), and by examples alone. *Same search* takes K labelled
matches as the examples and scores the rest, leaving out photos shot within two
minutes of one (a burst sibling would flatter them); *other site* takes them
from a search with the same words in folders that do not overlap. Either way an
example saved again is never scored (see **Leaks**, above).

## Calibrating "Same view over time"

Same view's cuts are set from your judgements the same way. *Label for
calibration* on a same-view run (or `calico eval new-view REFERENCE FOLDER`)
samples about 60 photos band by band of what the task uses: their patch share
with the reference (`evaluation.PATCH_SAMPLE_BANDS`: both tiers, just below them,
a few just outside the shortlist, two from anywhere), so the labels show where
same views stop, not only that the closest are right, and a repeat the
whole-photo match ranks low is still shown. Without patches it samples by the
whole-photo match (`SAME_VIEW_BANDS`).
You mark each *Same view*, *Nearly* (the camera moved) or *No*. `calico eval
same-view` then reports, per search and over all of them, AUC and precision
and recall at the current default, and the cut with the best balance: one cut
has to serve every view. With `--patches` it scores by patch share (the sure
cut), as the task does; otherwise by the whole-photo match of
`selection.view_match`. A frame saved twice (a re-export, a HEIC also kept as
JPEG) counts once, with the higher label if its copies differ
(`evaluation.count_copies_once`); frames of a burst each count.

The share ignores *where* the matches are, so a room with the same windows elsewhere can
share as much as the view itself. `--geometry` measures, against the share, only the
matches that agree on one placement of the reference: a shift, a zoom and a small turn on
the patch grid, fitted by RANSAC (`selection.patch_match.consistent_shares`, audit
A10-7). It reports over every labelled photo, and where the share is least sure
(`GEOMETRY_BAND`, 0.12 to 0.30), paired by reference. Measured, it was no better, so the
task does not use it (see "Decisions worth knowing"). A reference that is
not in a collection (an upload) is copied beside the labels, since uploads are
pruned after a week.

## Learning your taste (the Yours template)

*Rate for your taste* on the Label page (or `calico eval new-taste FOLDER`)
draws 60 photos at a place by value, one frame per burst (`evaluation.sample_by_value`,
0003): 15 at random from the whole place,
then General's first picks that Yours would not show (15), Yours's that General would not
(15), both (5) and the rest of the top quarter (10); turns a stratum cannot fill pass to
the next. First picks are what Strongest would show first there (`VALUE_DEPTH`, 40), each
as its burst's frame in the session; without a trained Yours, General's alone, and the
set's scorer says so. Sessions until 31 October 2026 drew 15 from each quarter of today's
general strength. You mark each *Like*, *Not sure* (left out) or *Don't
like*. Compare Strongest ratings and 👍/👎 on Strongest results also count: each photo
is labelled by its newest clear answer (like or don't like) across the three
(`reviews.taste_labels`), *Not sure* is left out, and `calico eval taste` names the rule
and counts the photos left out as unsure and those whose use cases disagree. Ratings
inside stories judge how a photo fits the story, not the photo, so they are left out.
Every label Yours learns from is one of your answers (a test holds it to that).

A second session at a place asks about other photos: a burst with any answer the taste
rule reads (*Not sure* included) is left out of the draw whole, so a rated photo's
near-twin is not drawn in its place (`reviews.reviewed`, passed to the sampler as
`exclude`), and a place with no more left than a session is taken whole, shuffled. Each item
records its stratum and the chance an unrated photo there had of being drawn
(`LabelItem.stratum`, `.inclusion`); neither reaches the page you rate on. `calico eval
taste` reads the stratum through the answer's session (`TasteLabels.session`): a photo drawn
for its value (`VALUE_STRATA`) counts among the first picks and is learned from, but the
AUCs over the whole range leave it out (`TastePhoto.whole_range`), since it would weigh the
top of a place more than the rest; each place says how many (`TasteRow.for_value`). So
does a photo whose answer came from Compare Strongest (owner's decision, 7 November
2026): it was rated because General or Yours showed it first, so judging a candidate on it
favours the tastes that chose it. That was 531 of 2,071 session ratings, liked 82% against
72.5% in taste sessions. AUCs measured before that day included them, so they are not
directly comparable with later ones. A place with nothing left to rate
is refused with a message that says so. Anchor sessions do the same with anchor answers,
each kind of photo a stratum.

`calico eval taste` judges each place rated in a session with a taste learned
from every other rating (`evaluation.measure_taste`): how often a photo you liked
outscores one you did not, by strength and by the aesthetic alone, General
against Yours, per embedding (`--embedder`), regularisation (`--l2`) and quality's
share of strength (`--quality-weight`). Strength is scored as Strongest ranks it, over each
location's whole pool (`evaluation.Pool`, 0011),
except for a candidate or a fit, which is read for the rated photos alone and compared with today's
the same way (the report says which). The photos a taste's ranking chose (drawn for their value,
shown in Compare Strongest) are left out of those AUCs and read apart in one line, Yours against
General, with what that means (0012);
`--scores FILE` measures a file of one number per photo beside them the same way, on the photos it
scores. Each report also gives strength minus the
aesthetic alone, paired, for General and Yours: whether mixing in quality at that weight
ranks your ratings better. Your ratings hold few of the clearly blurred or badly exposed
frames quality is there to keep off the first page, so a weight that costs nothing here
is a reason to lower it, not to drop it. `--technical 3` adds General's strength with
the noise-aware sharpness against today's, paired. `--fit centred` adds Yours fitted on
each photo's vector minus its place's mean over every catalogued photo there
(`evaluation.TASTE_FITS`, audit A10-4), against today's, paired; `--fit offset` adds
Yours with General's view as one more input that is not held back, so it learns only what
General misses; `--fit leaning` adds Yours leaning on General where your ratings hold few
photos like the one scored (audit plan S6): General, calibrated on the same ratings
(`models.taste.fit_general`), and Yours are mixed in log-odds by how familiar each photo is
(`models.taste.familiarity`: its `TASTE_NEIGHBOURS` (10) nearest rated photos against how
near rated photos are to the ratings at other places; at `FAMILIAR_AT` (0.5) of them or
nearer, Yours alone). Each
other fit is shown place by place, and collection by collection when you rated in more than
one, so a fit for trips is seen not to cost the abandoned places. `--fit by_scale` adds
Yours learning how much each scale is liked apart from what makes a photo strong within
its scale (`models.taste.fit_by_scale`; scale read with the built-in `scale` vocabulary,
so it needs a text encoder), and says per scale where the photos you liked rank.
`--blend` measures the candidate for General (`--aesthetic` or `--prompts`) averaged with
today's General by rank within each place, half each and untuned, so General stays apart
from your taste. With a candidate General, `--fit offset` reads it rather than today's:
Yours on top of the General being tried. Each candidate General and each other fit is
also judged at the top, by the share you liked among the top `TOP_SHARE` (0.25) of each
place's rated photos by strength, paired against today's: Strongest's bar is about what
it shows first, and a candidate cannot run Strongest here. `--aesthetic laion_aesthetic` adds General with LAION's aesthetic predictor
as its aesthetic (`features.laion`, audit A10-6), against today's prompts, paired, by
strength and by the aesthetic alone; like `--technical`, it is computed for the rated
photos alone and cached under its own key. It also
counts the photos you rated among the first 20 and 40 each template would show
there (`TASTE_FIRST`), and how many of them you liked. Strongest runs on the whole
place as a search would (bursts, one pick per view, diversity), but in memory
(`Sessions._first_picks`), so measuring saves no runs. One pick per view compares a place's
views part by part as a search does, so the first measurement after a place was rated
compares what no search there has, minutes at a large place, shown on stderr as "Comparing
views at PLACE (once per place)" (`sessions.PlaceProgressFn`); later ones reuse it. Sessions until 31 October 2026 sampled the whole range, so
few of their photos fall among the first picks; the counts are printed with every share.
`calico eval disagreements` lists the rated photos General and Yours rank furthest apart,
at least `DISAGREEMENT_GAP` (0.25) of a place's rated photos apart, each place judged by a
Yours that never learned from it (`evaluation.taste_disagreements`), grouped by whom you
sided with; `--sheet FILE.html` writes them as a contact sheet with the thumbnails inside,
and is refused inside a collection. `eval taste --prompts FILE` adds General with its
aesthetic worded as FILE says (the format of `models/taxonomies/aesthetic.toml`) against
today's wording, paired, scored zero-shot from the stored embeddings so no photo is read and
no search changes; each wording is a try of its own. Candidates live in
`docs/experiments/aesthetic/`, each today's wording with one prompt added, and move to its
`measured/` folder once measured.
`calico train taste` fits `models.taste` on every rating and saves it in
`models/` (`trained.py`) only if it is shown to beat General by strength: the paired
difference's interval lies above zero, at `WIN_PLACES` (5) judged places or more
(`TasteReport.yours_wins`). Otherwise it says Yours is "not shown to beat General", and
`--force` saves it anyway. It saves the places it learned from, and what a probability means:
`models.calibration` fits a slope and a bias on the logit of Yours's p(you like it) from the
forecasts of the measurement it has just made (each judged photo as a model that never learned
its place scored it, `TasteReport.forecasts`), of the photos a taste showed first (Compare
Strongest, the taste sessions' first picks: the photos a pick is), with `MIN_FORECASTS` (200)
of them at `MIN_PLACES` (5) places or more, and only if the probability clearly rises with the
model's.
It is kept with the taste it was read for, and read again at every training. A search chooses its template (`template`; default in `calico.toml`);
Yours falls back to General, with a note, until a taste is trained on the
embedding the photos have. Tasks never see the difference: `WithTaste` answers
"aesthetic" with the taste, and `Task.uses_taste` tells the composer where the
choice matters (Strongest, Stories).

## Rating anchors

*Could it anchor a story?* on the Label page (or `calico eval new-anchors FOLDER`)
asks, photo by photo, whether a photo could be a story's middle. It samples one frame
per burst, best first by General's strength, with each kind of photo taking its turn
(`evaluation.sample_for_anchors`: the `scale` label, and whether `subject` reads a
person or animal) and spread over strength within a kind: strength alone reached 2 of
your 28 finished stories' anchors. Without those vocabularies (an embedder that reads
no text) the sample is spread over strength alone and the set's `scorer` says so.
`calico eval anchors` (`evaluation.measure_anchors`) reports, for each signal Calico
has (general and Yours strength, a person or animal, a wide view, and the anchors
stories told by roles try first), how often a photo you could anchor with outscores
one you could not at the same place. A session of stories cannot judge anchors (a story
may work with its anchor and an outer photo swapped), which is why anchors have their
own. If no signal does well, anchors need a model of their own, trained on these
ratings as Yours is on taste.

## Compare Strongest

*Compare Strongest* on the Label page (or `calico eval new-compare FOLDER`) runs
Strongest at one place under General and under Yours (`COMPARE_PICKS`, 30 each),
each through `Engine.select`, so each is a saved run. It refuses, saying how to fix
it, until Yours is trained on the embedding the photos have. A photo both chose is
shown once (`evaluation.photos_to_compare`) and shuffled. Each `LabelItem` keeps
`shown_by` (template, run, rank), which the API leaves out, so the rating cannot be
swayed. Photos are rated *Like*, *Not sure* or *Don't like*, as in a taste session.
`calico eval compare` (`evaluation.measure_compare`) counts liked of rated among
each template's first `BAR_AT` (30) picks (the bar's own figure: a session that shows 60 does not count
twice the photos), and among the photos only one of them chose, with each session's pool size in bursts
(`collapse bursts` in Yours's saved run); `--min-bursts N` reports the places under it apart, since no size
is built in (0015). It reads each
list in tens (`CURVE_STEP`, 10) by the place each photo had in it: the liked share of the first ten,
the next, the last, and how far it falls between the first and the last, with an interval by
place. Where Strongest shows the strongest first the share falls (Yours, 14 fresh sessions,
10 November 2026: 0.90, 0.82, 0.75); a list made before 11 November 2026 was shown in the
order variety chose, and is flat. It also gives the liked share of each list's first `k` at each
`k` of `PRECISION_AT` (10, 20, 30, 60), at the places where both lists reach it, so a list of 30 has no 60
(`eval new-compare --picks` shows more of each list), where the share of the picks so far first falls below `RUNNING_LINE` 0.8 (read from
the `RUNNING_AFTER` 10th rated pick), and whether the line Yours drew where its list stops being good
sits there (0014). A session records
the taste that chose Yours's picks and whether it had learned from that place
(`LabelSet.taste`); sessions where it had, or where the taste did not record its places
(trained before 29 October 2026), are reported apart as in-sample, since there Yours is
judged on what it learned. A session made with `--cross-fit` (`Sessions.cross_fitted_taste`,
`Engine.select(taste=)`) shows Yours learned from every current rating except the hold-out's and
those at the place, the places next to it and copies of its photos, so a place the saved taste has
learned is fresh to it; it needs ratings, not a saved taste, is never saved, says so in the run's notes,
carries `TasteSeen.cross_fit`, and is reported apart (`CompareReport.cross_fitted`) from both the fresh
and the in-sample ones (0017).
A photo rated in two sessions counts once, in the newer, for
each state of the taste (learned the place, or not), so a later session at a place the taste
had learned never takes its photos from an earlier one it had not. The
ratings also count for your taste: `calico train taste` learns from them and `calico
eval taste` judges with them.

**Control photos** (`evaluation.controls`, 0008).
`calico controls draw` sets aside, once, a pool of `CONTROL_POOL` (40) photos rated in Compare
Strongest sessions, two thirds of them liked, at most 2 to a location and none the hold-out reaches (its places, the locations
next to them, and frames also filed under them; 0009)
(`<home>/controls.toml`). Each Compare session made after that mixes in
`CONTROLS_PER_SESSION` (10) of them, the least asked so far, none the session shows nor from a
location it shows (`LabelItem.control`; shown by neither taste, and the API leaves the flag out
like `shown_by`).
A control photo is answered as a review of its own (scope `{"control": "yes"}`): it does not
replace the answer it repeats, and it counts in no taste, share, training or draw.
`calico eval retest` reads the answers: how often you answer as you did when the pool was
drawn (liked then, liked again), how consecutive asks of one photo agree, each session's controls
liked (the same photos, so a fall is your standard, not a harder location), and the share liked a
perfect ranker's first picks could reach if the noise were symmetric.


## A session from a list

`calico eval new-list FILE` makes a taste session of the photos a CSV file names in its
`photo` column (`Sessions.label_list`). The photos are given, not drawn: it exists to rate,
blind, the photos two versions of a task disagree on. Each photo is shown once, in a
shuffle seeded by the photos alone, and the session keeps only the photos: which version
picked which stays with whoever wrote the list. A path the catalog does not hold is
refused by name. Each item's stratum is `evaluation.GIVEN`: the ratings are taste ratings
that `calico train taste` learns from, and since a ranking chose the photos, the
whole-range measures of `calico eval taste` leave them out, as they do Compare Strongest's.

## Rating stories

*Rate stories* on the Label page (or `calico eval new-stories FOLDER`) runs the story
task at one place four ways: General and Yours (Yours only once trained), each today's
way and the variant under test (`STORY_VARIANT`), `STORY_SESSION_STORIES` (6) stories
each. From 4 November 2026 the variant is `likely_anchors`: anchors tried by the anchor
prior (`anchor_by="likely"`); on 3 November it was `deeper`, 1,000 candidates rather
than 500, rated at seven large places and the default since 5 November. Sessions until 26 October 2026 told stories with
variety and without; those of 27
October today's way and *arranged* (anchors taking turns among wide exteriors, wide
interiors and portraits); those of 28 October to 2 November today's way and by roles
(`ROLE_STORIES`, `tell_by="roles"`). `StoryArm.way` records which (files of 27 October
saved it as an `arranged` flag, read as the way). To rate another variant, name it in
`labels.StoryWay` and set it in `STORY_VARIANT`.
To rate a named candidate instead (such as the improvement loop's, before it ships),
`calico eval new-stories FOLDER --variant NAME --template yours -o KEY=VALUE …` tells
today's way and the candidate's options under that template alone,
`CANDIDATE_SESSION_STORIES` (12) stories each; its arm's way is `candidate:NAME`
(`labels.CandidateWay`) and the set's `scorer` keeps the options.
The same three photos told two ways are one story to rate, credited to both, and the
stories are shuffled (`evaluation.stories_to_rate`). The page never sees how a story
was told; the set keeps it (`StoryItem.arms`), with the story's evidence, so it can be
re-scored without its run. Each telling is an ordinary saved run, so `calico runs
explain` works on it.

You mark each story *Works*, *Not sure* (left out) or *Doesn't*, and any photo that
alone does not belong *Doesn't fit* (a `story_fit` answer, never a thumb). `calico eval
stories` counts a session as one pool: works over doesn't, re-scored with and without
variety, and a table of how each way's stories were rated; 👍/👎 on story results are a
second report beside it. The six sessions told with and without variety confirmed
variety (AUC 0.46 without, 0.59-0.71 with); arranged stories worked less often than
today's (17 and 22 of 60). A session judges three photos together, not which should be
the anchor: a story may work with its anchor and an outer photo swapped, so anchors are
not judged from these ratings.
Story ratings never train your taste (see "Decisions worth knowing").

## Ratings and reviews

Every answer you give is one line in the ratings journal, `ratings/YYYY-MM.jsonl`
(`ratings_store.py`), filed in the month it was written, and never rewritten: a line is
only ever added, under a lock, in one write, so two Calico processes cannot lose each
other's answers, and every answer keeps its own time. Label set files are written once
and hold only the session: what was sampled, in what order, and what showed it.

An answer belongs to one **review**, `(use, scope, subject)` (`reviews.py`):

| use | the question | scope | subject |
|---|---|---|---|
| `describe` | does it match the text? | the text | a photo |
| `same_view` | is it the same view? | the reference | a photo |
| `taste`, `compare` | do you like it? | — | a photo |
| `anchor` | could it anchor a story? | — | a photo |
| `story` | does the story work? | — | its three photos, anchor in the middle |
| `story_fit` | does this photo fit the story? | — | the story's photos, then the photo |
| `result` | 👍/👎 on a result | the run's task and template | a photo |

The newest answer to a review is its rating; older answers to it stay as history, and an
answer never replaces another review's. So *Not sure* in a taste session replaces only
that photo's taste rating; a 👍 on a General Strongest result and a 👎 on Yours's are two
reviews; a photo rated in two taste sessions has one taste rating, the later one, shown
on both pages; and clearing an answer withdraws it. Each evaluation counts a review once,
in the newest session that asked it. Beside each answer is what showed it (template,
run, rank, way of telling) and the taste, kept out of the API in blind sessions.

Ratings were kept in the label set files and the catalog's `feedback` table until 29
October 2026. The first time a Calico opens a home it migrates them into the journal
(`ratings_migration.open_journal`, under a lock): every labelled photo as its set's
use, at the set's time (`approximate_time`); every rated story, and each photo marked as
not fitting as `story_fit`; every 👍/👎 as `result`, scoped by its run (the task unknown
when the run cannot be read). It notes `.migrated` only when every answer is in, so one
cut short is completed next time. `calico ratings migrate` does the same and prints the
counts against the sources. Nothing reads `feedback` but the migration; it is kept as a
read-only legacy source until a later catalog migration drops it.

## Web interface

`frontend/` is a React + TypeScript + Tailwind single-page app (Vite, TanStack
Query, React Router), built into `src/calico/web/static/dist/` and served by
FastAPI next to the JSON API. Its TypeScript API types are **generated** from
the server's OpenAPI schema (`make api-types`); a Python test fails if
`frontend/openapi.json` drifts from the server, so the two cannot silently
disagree. The page also carries a fingerprint of that schema
(`frontend/src/api/fingerprint.ts`, generated too); `/api/health` reports the
server's, and when they differ (a `calico serve` started before the code
changed) the page asks for a restart instead of failing requests with field
errors. Node comes from uv (`nodejs-wheel`), so there is still one toolchain.

## Extending

| To add | Do this |
|---|---|
| A **vocabulary / zero-shot model** (no code) | Drop a TOML file in `src/calico/models/taxonomies/` (ships with calico) or `<calico home>/taxonomies/` (personal). It is a `LabelModel` named after the file; a personal file replaces the built-in one of the same name. Only names a task reads change picks: `mood`, `place_type`, `surface` (story connections), `shot_kind` (kinds of shot), `aesthetic` (General). |
| A **sub-model** | Add labels `parent:child` to a taxonomy. `rollup` gives the parent view. |
| A **trained model** | Implement `ScoreModel` or `LabelModel`; register under `calico.models` in `pyproject.toml`. Same name as a taxonomy = replaces it. |
| A **feature extractor** | Implement `ImageExtractor`; register under `calico.extractors`. Bump `version` whenever its output changes. |
| A **task** (new menu entry) | Subclass `tasks.base.Task`; register under `calico.tasks`. The CLI and web form are generated from its `options_model`. |
| A **way of reading descriptions** | Implement `criteria.CriteriaScorer`; add it to `criteria.SCORERS`; compare with `calico eval criteria`. |
| A **rating session** or **evaluation** | Add `label_<kind>` / `evaluate_<kind>` to `sessions.Sessions`, built on the engine's public steps (`select`, `feature_view`, `template`, `patch_scores`); the measuring itself goes in `evaluation.py`. `engine.py` never imports `sessions.py`, and `tests/test_layers.py` keeps it so. |

## Observability (for humans and agents)

* `calico doctor [--json]` — environment, embedder and device, collections
  (mounted?), catalog size, feature coverage per extractor version, failures.
* `calico eval ratings [--legacy feedback.csv] [--quality-weight W]` — how well today's
  strength, quality and aesthetic agree with your 👍/👎 on Strongest and past stars, within each location
  (`ratings.py` puts the scales on one 0-1 scale).
* `calico eval new-view REFERENCE FOLDER | same-view [--json]` — label and measure same-view
  matches, and the `min_similarity` they support (see "Calibrating 'Same view over time'").
* `calico eval list | new TEXT FOLDER | criteria [--embedder NAME] | examples [--k K] [--json]` — labelled
  searches, and how each criteria scorer and text model ranks them (see "Measuring 'Describe it'").
* `calico ratings migrate [--json]` — copies every rating in the label sets and the catalog's
  👍/👎 into the ratings journal (`ratings/`, `ratings_migration.py`) and counts them again
  against those sources; it only reads them, and running it again adds nothing.
* `calico backup FOLDER [--with-catalog] | --verify FILE` — what cannot be recomputed (ratings,
  labels, runs, the trained taste, settings, the uploads runs point to) in one archive with a
  manifest; `--verify` reads every file back (`backup.py`). `calico doctor` warns when the
  newest backup is over 30 days old or more than 300 ratings are newer than it (10 November
  2026: a 12-day-old backup had missed 39.5% of the ratings), and names the folder the last one
  went into.
* `calico holdout draw | show [--json]` — set aside, once, the ratings no tuning may see;
  every `calico eval` leaves them out, and `--holdout` reports on them alone.
* `calico controls draw | show [--json]` — draw, once, the pool of photos Compare sessions ask
  again; `calico eval retest [--json]` reads what you answered.
* `calico locations [--json]` — every location name with its photo count,
  likely duplicates (typos, capitals, bracketed notes, names cut short), words
  spelt several ways, and folder names with stray spaces. Lists files only.
* `calico sandbox DIR` — a synthetic archive + isolated home. Try any change
  end to end here, never against a real archive or the user's calico home.
* `calico runs list | show RUN | explain RUN NAME` — any past run, and every
  recorded signal for any photo in its pool, including ones that were *not* picked.
* `calico select ... --json` — the full `SelectionResult` for scripts and agents.
* `calico tasks` — every task's options and their defaults, read from the code: the
  source of truth for weights this file describes only in words (`best`'s `diversity`,
  `diversity_reach`, burst gap). `calico models` — which models can run here.
* The rest (`index`, `collections`, `config`, `feedback`, `train`, `serve`) are listed by
  `calico --help`, each with its own `--help`.
* `-v` / `-vv` / `--log-json` on any command — decisions and timings on stderr.
* Web: `GET /api/health` (engine starting/ready/error), every run page has a
  *Diagnostics* panel (stages, counts, timings, versions).
* Browser tests (`pytest -m browser`) drive the real UI in Chrome; set
  `CALICO_SCREENSHOTS=DIR` to save screenshots of the pages they visit.

## Decisions worth knowing

Decisions from 30 October 2026 on are records of their own in `docs/decisions/`, with
their evidence; this table holds the earlier ones and is not rewritten.

| Decision | Why |
|---|---|
| Content key = BLAKE2b-128 of the whole file | Features are shared by identical bytes, so the key must never collide; a head+tail hash can. We read the file to decode it anyway. |
| Paths are NFC-normalised at the scanner and catalog (folder *and* file names), and every path from outside is spelt as the disk spells it (`calico.paths.normalise`) | macOS hands out NFD and NFC for the same name; without this one file becomes two rows, or flips in and out of the catalog on alternate runs. (On Linux, NFD-named files may not open.) A folder in a request, a collection root, a command line, or a place alias can arrive typed in other capitals (macOS volumes ignore case), in NFD (pasted from Finder) or through a link; each is resolved, matched to the entry that is the same file, and NFC'd, so it is the catalogued folder rather than "not inside any collection" or no photos. Audit 28 October 2026, A01-3 and R-07. |
| A lone RAW is read as the JPEG its camera embedded (LibRaw, `rawpy`), developed at half size only when that preview is under 2048 px | The preview is the camera's own rendering, as a JPEG beside it would be, and LibRaw writes the camera, capture time and orientation into it, so the RAW is catalogued like any JPEG. A Canon CR3 embeds one at full size (0.2-0.3 s a file); a DJI DNG's is small, so it is developed (about 0.2 s). A RAW beside its JPEG stays represented by the JPEG. |
| Symlinked files and folders are skipped (and counted in a run note) | A link can point outside every collection; the archive is defined by real folders. |
| The web app answers only this computer, only requests addressed to it, and changes only from its own pages | It has no login. So `calico serve` listens only on a loopback address (127.0.0.1, ::1, localhost), and the app refuses any connection from another computer whatever its headers say. A device on the network could otherwise browse every photo and delete labels; to use Calico from one, tunnel in with `ssh -L 8765:127.0.0.1:8765`. Any web page the user visits could otherwise drive it (a form post: Origin must be the server's own scheme, host and port, and `Sec-Fetch-Site` not another site), read it (DNS rebinding: Host must be loopback) or frame it (`frame-ancestors 'none'`). Audit 28 October 2026, R-02 and A06-1. |
| Derived fields (location, collection) are recomputed on every index | They come from configuration, which can change without the file changing. |
| A signal that cannot be measured is left out and its weight shared, never filled with a constant; and every photo left out says why | Telling by roles without the scale model cannot tell `fresh` (one space at one scale), nor `anchor` without subject or scale; they leave the evidence and the measured roles share their weight, so no story is credited "not three shots of one space" unchecked. Options roles do not read are named. `calico runs explain` gives the first reason a photo was not picked: no capture time, a copy counted once, not the kind asked for, a stronger frame of its burst, its strength rank, or in no story chosen. Recurring views finds look-alike pairs block by block and groups each cluster of compared photos alone (no n x n matrix), says how many pairs it will compare and about how long, and above `max_pairs_per_place` compares each photo with its closest look-alikes only. Audit A03-1, A03-2, A03-5, A03-6, A03-9, A04-6, A10-8. |
| Collections may nest; a photo belongs to the deepest root, from its first index, and each folder follows its own collection's exclusions. The calico home may be inside no collection, nor hold one | A nested collection gives a part of the archive its own location rule. A home inside the archive would put the catalog and thumbnails into it (the archive is read-only) and index them as photos, so `Engine.open` and `calico index` refuse it, while `calico config` and `calico collections` still run to fix it. `calico collections remove NAME` forgets a collection without touching its folder. Audit A02-3, A02-4, A07-12. |
| One photo never stops an index, and an index never stays quietly wrong | A photo an extractor cannot read is retried alone, recorded as `extract:<name>@<version>` and left out; an extractor that fails on every photo of a batch, or returns the wrong shape, stops the run (`ExtractorError`), since that is the model, not the photos. A catalogued file that changes and can no longer be read leaves the catalog. A missing thumbnail is made again. Files of other kinds and linked folders are counted and named. 16-bit and 12-bit greyscale scans are read by their real range instead of clipped to white (no extractor version changes: only images that were read wrongly change). A capture date before 1826 or after next year is a damaged tag, so the photo is undated; a historic scan keeps an 1880s date. Audit A02-2, A02-5, A02-6, A02-7, A02-10, R-06. |
| A place can be named by a deeper folder (`location_deepest`), passing over folders listed as not places | A travel collection is often filed by trip; its folders below the trip often name the city or site ("Coast Town, Farland", "Unedited/Calder"), mixed with folders that only organise ("Originals", "Lightroom Edits", "App"). Which names organise is a personal convention, so it is configuration (`location_not_places`), not a built-in list. GPS was the first idea, but only 1% of a travel sample carried it. |
| Location names: spacing normalised automatically, everything else by configuration | Merging similar names automatically would join different places ("Mill Pond", "Mill Road"). `calico locations` suggests; `location_aliases` and `location_ignore` decide. |
| Engine drops photos with non-finite features | One boundary instead of NaN checks in every task; tasks also sanitise third-party model output. |
| `quality` is pool-relative (sharpness percentile) | Absolute sharpness has no meaning across subjects; with very small pools it carries little information. |
| Sharpness takes off the noise only once your ratings show it helps (`technical@3`, opt-in) | Today's sharpness (Laplacian variance over grey variance) reads noise as detail, so a dark, blurred, high-ISO frame outscores a sharp one (audit R-05). Version 3 takes twice a noise estimate (the MAD of the Laplacian, `features/technical.NOISE_MARGIN`) off first: on synthetic frames every dark noisy blur then fell below its sharp frame, and clean frames kept their value. `calico eval taste --technical 3` measures General's strength with it against today's, paired, on the rated photos alone (cached as `technical@3`, so the archive is not indexed again). It becomes the default only if that difference's range lies above zero. Measured on the owner's ratings (542 rated, 11 places; 2026-10-29) it did not: +0.02 [-0.01, 0.04] at quality weight 0.2, +0.04 [-0.01, 0.08] at 0.35, +0.07 [-0.01, 0.13] at 0.5. On 975 ratings at 20 places (2026-10-30) the gain shrank: +0.01 [-0.01, 0.03] at 0.2 (the default now), +0.02 [-0.02, 0.06] at 0.35, +0.04 [-0.02, 0.09] at 0.5. On 1,611 ratings at 32 places (5 November 2026, a best of 34 settings tried): +0.01 [-0.01, 0.03] at 0.2. So it stays opt-in: `extractor_options.technical.noise_aware = true` in `calico.toml` (then `calico index` computes it for every photo). |
| Quality is a fifth of strength (0001), the aesthetic the rest (measured: at 0.35 it ranked your likes worse) | `eval taste` gives strength minus the aesthetic alone, paired. On 542 ratings at 11 places (29 October 2026) neither 0.2 nor 0.35 differed by more than chance, so 0.35 stayed. On 975 ratings at 20 places (30 October 2026), General: +0.00 at 0, -0.00 [-0.01, 0.01] at 0.1, -0.01 [-0.03, 0.01] at 0.2, -0.04 [-0.07, 0.00] at 0.35, -0.10 [-0.15, -0.03] at 0.5; Yours at 0.35, -0.04 [-0.07, -0.01]. On the four hold-out places at 0.2: General +0.01 [-0.02, 0.04], Yours +0.02 [-0.01, 0.04]. Quality tells your likes apart no better than the aesthetic does, and costs from 0.35 up. It keeps a share rather than none because taste sessions hold few of the clearly blurred or black frames it keeps off the first page, which these numbers cannot see. One pair of constants for every task (`selection.strength.QUALITY_WEIGHT` 0.2, `AESTHETIC_WEIGHT` 0.8); a search can still set `quality_weight`. |
| Yours is fitted across places, not centred on each (`--fit centred`, measured) | Audit A10-4 expected a taste fitted across places to learn "this kind of site", since places sit apart in every embedding and you like more at some. `eval taste --fit centred` fits on each vector minus its place's mean over every catalogued photo (`evaluation.TASTE_FITS`); Yours is linear, so only the fit changes, never an order within a place. On 542 ratings at 11 places (29 October 2026) it was worse on every embedding and weight: CLIP -0.04 [-0.09, -0.01] at quality weight 0.2 and -0.03 [-0.07, -0.00] at 0.35, SigLIP -0.05 [-0.10, -0.02] and -0.05 [-0.09, -0.01], DINOv2 -0.04 [-0.10, -0.01] at 0.2. On 975 ratings at 20 places (30 October 2026), CLIP at 0.35: -0.02 [-0.04, -0.01], its whole range below zero. What sets your places apart also tells likes within a place, so today's fit stays. |
| Yours does not lean on General where it has seen little (`--fit leaning`, measured) | Yours is learned mostly from abandoned places and ranks worse than General on travel, so it would mix in General, calibrated on the same ratings, where a photo's nearest rated photos are further than usual (audit plan S6; `models.taste.familiarity`). Set before measuring: kept only if travel is shown better and the abandoned places are not shown worse. On 1,611 ratings at 32 places (7 November 2026), CLIP: +0.00 [-0.02, 0.03] overall, travel +0.03 [-0.11, 0.11] at 9 places, explore -0.01 [-0.04, 0.02] at 18; DINOv2 travel +0.09 [-0.03, 0.19]. Travel moved up on every embedder, but 9 places cannot show it, so today's Yours stays. Run again when more travel places are rated. |
| Yours does not take General as an offset (`--fit offset`, measured) | With General's logit as one more input that `l2` does not hold back (`fit_taste(general=)`), Yours would learn only what General misses (audit A10-4). `l2` then also sets the mix, so it was measured at three, each against today's Yours at l2 1000. On 542 ratings at 11 places (29 October 2026), CLIP: -0.07 [-0.12, -0.01] at l2 10, -0.02 [-0.05, 0.02] at 100, +0.00 [-0.03, 0.03] at 1000. General is zero-shot on the same CLIP embedding, so its logit is nearly a direction Yours can already learn: nothing new to add. A search cannot score such a taste (`TasteModel` refuses it). |
| General's aesthetic stays zero-shot, not LAION's predictor (`--aesthetic laion_aesthetic`, measured) | Today's General compares a ViT-B/16 embedding with a few text prompts, so its score is nearly a direction Yours can already learn (the offset above). LAION's predictor is a head trained on about 240,000 human ratings (SAC, LOGOS, AVA) over OpenAI's CLIP ViT-L/14: a larger embedding and a direction learned from ratings, a view Calico does not have (audit A10-6). Its layers have no activations between them, so it is one straight line through that embedding, applied in numpy (`features.laion.head_score`); its weights come from a fixed commit of LAION's repository (Apache-2.0), are checked by SHA-256 and are loaded without running pickled code. On the Mac it matches LAION's reference network (`tests/features/test_laion.py`, ml). It replaces the prompts only if General's strength with it minus today's has its range above zero. Measured on 542 ratings at 11 places (30 October 2026): +0.03 [-0.05, 0.14] at quality weight 0.2, +0.02 [-0.05, 0.12] at 0.35, +0.01 [-0.04, 0.10] at 0.5; by the aesthetic alone 0.85 against 0.79, +0.05 [-0.03, 0.17]. On 975 ratings at 20 places (30 October 2026) the gain shrank: +0.01 [-0.05, 0.08] at 0.2, +0.00 [-0.06, 0.07] at 0.35, +0.00 [-0.06, 0.05] at 0.5; by the aesthetic alone 0.80 against 0.78, +0.03 [-0.05, 0.10]. On 1,611 ratings at 32 places (5 November 2026, a best of 34 settings tried): +0.04 [-0.03, 0.10] at 0.2; by the aesthetic alone 0.79, +0.05 [-0.02, 0.12]. Each range reaches below zero, so today's prompts stay. On 542 ratings, by the aesthetic alone it matched Yours (0.84) without learning from any of your ratings. Adopting it would also mean computing ViT-L/14 for every photo (`calico index`), about 1.7 GB of weights. |
| `layout` is a spatial pyramid of gradient-orientation histograms | Survives light and colour-cast changes (cos ≈ 0.99) and small re-aims (≈ 0.8) while separating different scenes (max ≈ 0.62). Numbers in `features/layout.py`. |
| One worker thread owns the web app's `Engine` | SQLite connections cannot cross threads; selections are serialised, which is fine for one user. |
| "Describe it" is read into parts before CLIP sees it (`parts` scorer) | CLIP blurs a sentence into one meaning: "a sky without clouds" found cloudy skies first. Things to avoid become penalties; months, seasons and years use capture dates; a place after "at/in/from/near" uses location names. Measured on labels: precision@10 0.82 → 0.87, AUC 0.80 → 0.83, no search worse. |
| Light, colour and mood words stay with CLIP; measuring them is opt-in (`parts-looks`) | Whole-photo measurements (warmth, saturation) separated "golden hour" and "brooding" on the three labelled searches that suggested them, but on six fresh searches they did worse than CLIP (AUC 0.60 vs 0.74): golden hour shows up as saturation at one site, warmth at another, contrast at a third. They track light within a shoot, not across sites. |
| Text tricks were measured and not adopted | Prompt templates made ranking slightly worse; subtracting a neutral caption helped a little on average but hurt several searches and did not make scores comparable across searches. |
| "Describe it" is read by SigLIP (ViT-B-16, webli); everything else stays on CLIP | Same 30 labelled searches, same reading: precision@10 0.80 → 0.85, nDCG@10 0.79 → 0.85, AUC 0.81 → 0.86. The larger SO400M was no better. CLIP stays for same view, stories and kinds of shot because SigLIP has not been measured there. Each description is read as the caption "a photo of …." (AUC 0.840 → 0.847). |
| A clear match is SigLIP's own line (p ≥ 0.5), not a tuned threshold | Of 452 photos it called clear matches on the labels, 94% were labelled Matches and 1% No; whole folders without snow or bicycles had none. CLIP has no such line: its similarities shift with every description, and at its best cut-off it "found" snow and bicycles throughout folders that have neither. |
| No "nothing here matches" verdict, and "only clear matches" falls back to the closest | SigLIP is as unsure of vague descriptions it can see (golden hour, brooding: peak p ≈ 0.0002–0.006) as of absent things, so "no clear match" cannot tell the two apart. The note says so instead of hiding photos. |
| Times of day stay with CLIP | Camera clocks are often on another time zone (a cruise's photos read 3-5am), so the recorded hour cannot be trusted for "morning" or "golden hour". |
| Ambiguous words are only taken when unmistakable | "May"/"March" need a capital or "in"; "fall"/"spring" need "in"/"during"; a one-word place needs a capital, so "a wedding in church" never narrows to a place named "Church". |
| Settings files hold only what was chosen | Writing every default froze them in place, so a better default (like `parts`) never reached existing users. |
| Examples come from your collections ("More like this"), not uploads | Measured: three matches from the same search as examples lifted the rest of it (AUC 0.86 → 0.89, precision@10 0.81 → 0.84); examples from another site did not help (AUC 0.75 → 0.75, worse at higher weights), because they bring their own place along with its light. |
| One "how much the description counts" for every task, not a weight per task | Each task had its own `criteria_weight`, meaning three different things (relative to quality and aesthetic in `best`, a share in `same_view`, fixed in `triptych`). Now it is the same share everywhere, beside the description box; the old option is refused with a pointer to `describe_weight`. |
| Recurring views groups by average linkage, not by any matching pair | Joining whenever two photos match chained look-alikes: one group of 1,613 photos at a large site, 69 of 219 "not the same view" photos pulled in. Average linkage at 0.17 pulled in 2 and kept 64 of 69 labelled repeats with their reference. Patches are not stored compressed: PCA to 64-256 dimensions let 5-7 wrong photos past the sure cut instead of 3, and matched no faster. |
| Capture times stay as the camera's clock showed them; photos close on that clock *or* in UTC are one occasion | Abroad a phone moves to local time while a camera keeps the zone it was set to (on one visit: Canon −04:00, iPhone +01:00, five hours apart by their clocks, together in UTC). Both write `OffsetTimeOriginal`, kept as `utc_offset`. Months, seasons and titles keep the local clock, which is what was seen. A photo without a zone borrows its camera's in the same folder, else the folder's single zone; with two zones in a folder (the Canon and the phone abroad) an untagged camera keeps its own clock. Photos catalogued before this get their zone read again from EXIF on the next index, without decoding. UTC alone was tried first and split visits: a frame tagged with its zone and its export without one, in a sibling folder (`RAW/` and `JPG/`), landed 4 hours apart, and 213 of the first run's 500 views shown from one collection were only such splits. With either clock enough, 286 of those 500 stay, one fewer than by the camera's clock alone. |
| An edit is dated by its original only when it has no camera recorded | The first full run counted one frame as three occasions: two edits, cropped and recoloured (so not copies), carried their export dates. An edit that keeps its camera data keeps its capture date, so only camera-less photos are re-dated. Where one photo of a same-place pair 2+ hours apart had no camera and they shared ≥ 0.50, 19 of 20 (fresh sample) were edits; with a camera on both, 29 of 36 such pairs were the view shot again, so the rule never re-dates those. Heavy crops (shares 0.40-0.50) and edits that kept the camera but not the date are still missed. |
| Same view counts every mutual patch match, not only those in place (`eval same-view --geometry`, measured) | A share that ignores where matches are cannot tell a room with the same windows elsewhere from the view (audit A10-7). `consistent_shares` keeps only the matches that agree on one placement of the reference: a shift, a zoom and a small turn, fitted by RANSAC on the 32x32 grid. On 14 references, 591 labelled photos (2 November 2026): mean AUC 1.00 against 0.99 in place, -0.01 [-0.03, 0.00]; where the share is 0.12 to 0.30 (277 photos), 0.96 against 0.89, -0.06 [-0.17, 0.00]. A true repeat keeps only about a fifth of its matches in place (best cut 0.05, against 0.19), likely because between visits near and far parts of a view move differently: drone views lost most (one fell from 100% to 55% recall). It did lift precision where look-alikes get in (9% to 25%, 56% to 100%, 36% to 80% at three references), so a looser placement could be tried there; the hold-out has not been looked at. |
| Same view's cuts are final, and labelling for them has stopped | 19 references, chosen to cover nearly every view shot again across visits. More would be views the match already finds, which would bend the measurement toward the match rather than test it. *Nearly* (re-aimed) counts as welcome, not wrong. |
| General's aesthetic names architectural geometry, phone snapshots and reference shots (version 4) | Where General and Yours disagreed most (`eval disagreements`, 7 November 2026), the owner saw General miss strong architectural geometry, and rank phone snapshots and reference photos of objects and museum displays above portfolio-worthy photos. One prompt for each, added to strong, weak and document; each alone gave about +0.01 by strength (`eval taste --prompts`), and the three together +0.037 [0.005, 0.078] (AUC 0.739 → 0.777) on 1,611 ratings at 32 places, the largest change to General's wording so far. That was a best of 41 settings tried, so it was checked once on the hold-out (4 places, 205 ratings): +0.002 [-0.030, 0.040]. Kept on that direction; the size of the gain is not confirmed, and the hold-out places are ones where General already ranks well (0.835). |
| General's aesthetic has a `document` label | Signs, labels, pages and screenshots are sharp and evenly lit, so a zero-shot "strong vs weak (blurry, badly framed)" scored them as strong. A third label takes probability from "strong" without blurring "weak": General's strength AUC on 1,022 taste ratings 0.755 → 0.771, no place worse by more than 0.01; the words added to "weak" instead cost three places 0.03-0.04. |
| Diversity chooses only among the strongest (`diversity_reach` × the number asked for) | MMR rewards being unlike what is picked, so at a place full of rooms it promoted placards and a book cover that scored among the weakest. Capping its reach keeps diversity to choosing between strong photos. |
| One camera's frames burst across folders | An edit filed beside its original in another folder ("Lightroom Edits", "iPhone X") keeps the camera and time, so Strongest showed both (6 pairs in one site's top 40). With no camera, two folders may be two people, so they still burst only within one folder. A CLIP near-duplicate cut regardless of time would also have merged re-shoots 10+ minutes apart (53-156 pairs per place). |
| Screenshots are left out per collection, not everywhere | A PNG with no camera is a screenshot at phone sizes, but in ARCHIVES it is a historic scan; whether to leave them out depends on the collection (`exclude_screenshots`). |
| Stories weigh variety, judged on the set's mean similarity | Every story you rejected outright was three frames of one moment (every pair 0.80-0.94 by CLIP); stories you liked averaged 0.79 and could hold one pair at 0.92-0.95 when the third photo differed, so a tighter near-duplicate cut on pairs would have dropped liked stories. Liked over rejected (`calico eval stories`, 16 and 5 stories): AUC 0.67 → 1.00 at `too_alike` 0.80, `variety_weight` 0.25. Lowered to 0.70 after the finished stories (0.62 alike, Calico's 0.79 at 0.80): still 1.00 on the ratings, and Calico's stories fall to 0.71-0.72 alike. Six rated sessions at other places confirmed it (26 October 2026): 0.46 without variety, 0.66-0.71 with it from 0.70 to 0.85. |
| Stories consider 1,000 candidates and connect by place type too | Half of the 26 stories you had finished then used a photo outside a place's 150 strongest; 500 let 77% (General) and 69% (Yours) be told, keeps the picks strong (anchors are the strongest) and costs under half a second. 1,000 (from 5 November 2026) lets 86% and 82% of 28 be told, against 79% and 71% at 500, at 9.3 s a search against 9.1 s at one large site; in seven rating sessions at large places its stories worked 19 of 42 (General) and 20 of 42 (Yours), +0.02 [-0.21, 0.27] and +0.10 [-0.14, 0.36] against 500 in the same sessions: not shown better, but not worse, which was the bar set for it. Below about 900 photos the two tell the same stories. Place type is what most sets finished stories apart from random sets of three (0.64 vs 0.45); as a connection it raised how often they beat a swapped story at 13 of 26 and lowered it at 8. Candidates by kind of shot and `kind_mix` by default were tried and not kept. |
| Telling by roles is rated in sessions; anchors by kind, the strongest connection and variety by repetition are choices only | Your 28 finished stories are anchored on wide shots or portraits, Calico's on its strongest photos (nearly all wide interiors; no drone shot at one large site anchored a story though all were candidates), so `anchor_kinds` lets kinds take turns; the swap test cannot judge anchors, so *Rate stories* does. `connect_by="strongest"` and `variety_by="repetition"` (one space by the view embedder at one kind of shot) came from the same ratings but made your finished stories beat a swap less often (0.71 → 0.66 each, 0.61 both, General): kept to measure again once the features see what you mean. Arranged stories worked less often than today's in five sessions (17 and 22 of 60). *By roles* (`ROLE_WEIGHTS`: a close pair, the third photo belonging, an echo or a change of `scale`, not three shots of one space at one scale, an anchor with a `subject` or the widest view; anchors tried by half strength, half person-or-wide) raised how often your stories beat a swap, 0.73 → 0.75 (General) and 0.70 → 0.72 (Yours), so sessions rated it from 28 October to 2 November 2026: 36 of 107 of its stories worked, against 48 of 105 today's way, in the nine sessions told with `distinct`. Not shown to be worse (a story told both ways counts for both), but it did no better anywhere, so its place in sessions went to the next variant (`STORY_VARIANT`). |
| Ratings are kept by review, in a journal | A 👍/👎 recorded neither which template nor which task showed the photo, story reviews turned their "doesn't fit" marks into thumbs, and any session verdict beat any 👍/👎, the oldest session winning (audit 28 October 2026, A05-1, A05-10). A rating now belongs to one question about one thing, its newest answer wins, and nothing converts one use's answers into another's. The journal only ever adds lines, so no write can lose another process's answer, and each keeps its time. |
| Story 👍/👎 do not train photo taste | Asked: a 👎 inside a story meant "doesn't fit the story", not "weak photo". They are kept for learning story connections. |
| Taste is rated in sessions across the whole range, not only on results (superseded by 0003: sessions still sample beyond results, but mostly first picks) | Ratings on results are on photos Calico already ranked high: of 147, 28 were dislikes, too few to learn or test what you dislike. Sessions sample evenly over today's strength. |
| Yours learns on CLIP, held back hard (l2 1000) | Measured on 853 ratings, each place judged by a taste that never saw it: strength AUC 0.76 (General) → 0.83 (CLIP), 0.82 (SigLIP), 0.78 (DINOv2, at best). With light regularisation it learned the rated places by heart (0.74 at l2 0.1); from 300 to 3000 it was flat. Its probabilities sit near 0.5, which does not matter: strength uses their percentile. |
| `tiny` embedder exists | Calico works without PyTorch, and tests run in seconds. It cannot read text; the engine says so whenever that matters. Tests set `embedder`, `describe_embedder` and `view_embedder` to it. |

## Data on disk (`$CALICO_HOME`, default: platform user-data dir)

```
calico.toml          settings: collections, embedders
looks.toml           your light, colour and mood words, replacing built-in ones by name (optional)
catalog.sqlite       photos, features (keyed by content + extractor version), failures, pair shares
                     (recurring views: two photos compared part by part); can be rebuilt by indexing.
                     Its feedback table is the 👍/👎 kept before the ratings journal, read only by
                     the migration
patches-<pid>-*/     scratch patches while a run compares pairs (deleted when it ends; one a killed
                     Calico left is deleted when the next starts, and calico doctor reports them)
thumbs/ab/abcd….jpg  thumbnails by content key
runs/<run_id>/       result.json, signals.csv
labels/<set_id>.json rating sessions, written once: what was sampled and what showed it (they cannot be recomputed)
labels/.trash/       deleted sets, with when they were deleted: move one back to undo
ratings/YYYY-MM.jsonl every answer you gave, one line each, in the month it was written: the one
                     folder that cannot be recomputed from anything (see "Ratings and reviews")
last_backup.json     when and where the newest calico backup was written (for calico doctor)
holdout.toml         the hold-out: places, same-view references and story sessions no tuning may
                     see (calico holdout draw), written once and never replaced
controls.toml        the control photos: a pool drawn once from Compare ratings and how each was
                     rated (calico controls draw), never replaced
experiments.jsonl    every setting a calico eval measured, one line per call, so a best is read as a
                     best of N
models/taste.npz     your trained taste, for the Yours template (calico train taste), with its calibration; taste.json beside it for older Calicos
taxonomies/*.toml    personal vocabularies (optional)
uploads/<id>/        reference photos sent to the web interface, deleted after a week
```

The archive itself is **never written to**.

What cannot be recomputed (ratings, label sets, runs, the trained taste) is written so a
crash or a second Calico process loses nothing (`calico/storage.py`): a rating is a line
added to the journal in one write, under a lock that a second process respects too, and a
line cut short by a crash is skipped and counted; everything else is written atomically,
so a reader sees the old file or the new one, never half of one. A new label set or run
never replaces one with the same id. `calico backup` keeps all of it in one file. A file this Calico
cannot read, written by another version or damaged, is left untouched and left out:
`calico doctor` names it, every `calico eval` and `calico train` says it is missing, and
`calico train taste` refuses to learn without it unless told `--accept-missing`. A label
set is never saved again once made, so one with a newer `schema_version` is read and rated
like any other. `calico sandbox --force` replaces only a sandbox it made (it
writes `.calico-sandbox`), never a folder holding the calico home or a collection.


## Continuous integration

`.github/workflows/ci.yml` runs `make check` on GitHub's hosted runners, `ubuntu-latest` and
`macos-latest` (where the tests of decomposed (NFD) names and of case run), for every push to
a branch and every pull request; Dependabot's branches run it through their pull requests
only. It reads the repository only; there is no self-hosted runner
and no deploy step (`tests/test_workflows.py`). Readers skip what an older version cannot
parse (`RunStore.recent`, `LabelStore.all`), and catalog migrations only add, so going back
to an earlier version never breaks on data a newer one wrote.

Both jobs run every test, the ML and browser tests included: `make check` calls `uv run`,
which installs every default group, PyTorch with it. CI fetches the model weights afresh, and
`tests/features/test_golden.py` compares one photo's output from each ML extractor with
`tests/features/golden.json` (cosine above 0.9999), so weights replaced on the hub under the
same name, which would mix old and new vectors under one cached `version`, fail CI. Each job
states how many tests it expects to skip (`CALICO_MAX_SKIPPED`: 6 on Linux, 2 on macOS), and
one more fails the run (`tests/skip_limit.py`): a runner that lost PyTorch or Chrome would
otherwise pass while testing less.

The macOS job runs the models on the CPU (`CALICO_DEVICE=cpu`, read by
`features.hub.auto_device`): GitHub's macOS runners report an MPS GPU, and CLIP's answers on it
were wrong (a plain blue image did not match "a plain solid blue image").
