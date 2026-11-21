# Calico

Pick photographs from an archive. Choose what you want from a menu, pick the
folders, optionally describe it in words or attach a reference photo, and get
back a gallery of picks — each with the reasons it was chosen.

| Menu entry | What it returns |
|---|---|
| **Strongest photographs** | The best frames in the folders, one per view (a view shot on several occasions shows its best frame, with the others behind it), overall, of one kind (establishing / detail / portrait), or matching your description |
| **Same view over time** | Every photo that closely matches a reference — the same room, object or view — in date order, so changes in season, light, decay or overgrowth read left to right |
| **Views you keep coming back to** | Without a reference: at each place, every view you photographed on two or more occasions (a different day, or the same day hours apart), most occasions first, one photo per occasion. The first run at a large place takes a while; what it compares is remembered |
| **Three-photo stories** | Sets of three that belong together: by default a shared theme, the same kind of place, matching light or mood, and matching colour; on request the same location, the same surfaces (peeling paint, brick, rust) or complementary kinds of shot. Give it an anchor photo and every story is built around yours |

It grows by adding **tasks** (menu entries), **models** (judgements such as
"what kind of shot is this"), **vocabularies** (a TOML file of labels — no
code) and **feature extractors**. See [docs/architecture.md](docs/architecture.md) for
how it fits together.

**Your photos stay yours.** Calico runs on your own computer and only ever reads your
archive. Its catalog, your ratings and the taste it learns from them live in your Calico
home on that computer; nothing is uploaded anywhere. The only things it downloads are the
model weights listed under [Third-party models](#third-party-models). The web interface
answers this computer only.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `make`. Clone this repository, then in it:

```bash
make setup               # environment (including CLIP/PyTorch), the web interface, git hooks
```

Without PyTorch (`uv sync --no-group ml`, or installing `calico` without the `ml` extra) Calico still works with the built-in `tiny` embedder
(composition and colour only) and tells you whenever that limits a request.
Models run on the GPU PyTorch finds (Apple's MPS, or CUDA), else the CPU; set
`CALICO_DEVICE=cpu` (or `mps`, `cuda`) to choose.
RAW files with no JPEG beside them are read through LibRaw (the `raw` extra, installed
by default); without it they are skipped and counted, and `calico doctor` says so.

## Use

```bash
uv run calico collections add photos "/Volumes/Archive/Photos" \
    --location-pattern '^(?P<location>.+?) - '   # optional: folder name -> location
uv run calico config set-embedder clip           # once the ml extra is installed
                                                 # ("Describe it" uses SigLIP by default)
uv run calico index "/Volumes/Archive/Photos"   # optional: requests index on demand
uv run calico serve                              # the web interface
```

or from the command line:

```bash
uv run calico select best "/Volumes/Archive/Photos/Old Mill - 2021" --criteria "light through broken windows"
uv run calico select same_view "/Volumes/Archive/Photos" -r reference.jpg
uv run calico select triptych "/Volumes/Archive/Photos" -o connections=mood,color --limit 5
uv run calico select recurring_views "/Volumes/Archive/Photos"   # once overnight: compares and remembers every place
uv run calico runs explain <run-id> IMG_1234      # why a photo did or didn't make it
uv run calico doctor                             # environment, catalog and feature coverage
uv run calico sandbox /tmp/cal                   # a synthetic archive to try things on
```

Calico never writes to your archive. Its catalog, thumbnails and saved runs
live in `$CALICO_HOME` (default: your platform's user-data folder).

Your ratings, labels, saved runs and trained taste cannot be recomputed, so back them up to
another disk now and then (`calico doctor` warns after 30 days, or after 300 ratings made since
the last backup; give `calico backup` a real folder, not the word `DEST` or `<folder>`):

```bash
uv run calico backup "/Volumes/Backup/Calico"          # calico-backup-<date>-<time>.tar.gz
uv run calico backup --verify "/Volumes/Backup/Calico/calico-backup-….tar.gz"
```

`--with-catalog` adds the catalog too (features and the index), which can otherwise be
rebuilt by indexing again. A restore is unpacking the file into an empty Calico home.

### Describing what you want

"Describe it" understands more than CLIP alone: *without …* / *no …* count
against a photo, months, seasons and years use capture dates (set
`hemisphere = "south"` in `calico.toml` if needed), and a place after *at*,
*in*, *from* or *near* uses your location names, e.g. "fog without people at
Old Mill in winter". The composer shows how it read your words.

Descriptions are read by SigLIP, which can say when it is confident a photo
matches: those photos get a *Clear match* badge, and *Only clear matches*
(`--only-clear-matches` on the command line) shows just them. It is rarely
confident about light or mood, so when nothing clears the bar you see the
closest photos and a note saying so.

*More like this* on any result (except a same-view timeline) starts the same
search again with that photo as an example: part of the description, left out
of the results (`--example PATH` on the command line). Examples help most with
looks words miss, like "brooding", within the folders they come from; from
another place they bring that place along, so Calico notes it.

The slider beside the box sets how much the description counts: its share of
each photo's score (`--describe-weight 0.7` on the command line). Each task
starts from its own default (half for the strongest photographs, 30% for the
same view, a third of each photo's strength for stories); `calico tasks`
lists them. At 0% the description only marks clear matches.

### Your taste

Strongest photographs and stories rank by *strength*, which includes an
aesthetic judgement. **General** uses Calico's zero-shot one; **Yours** learns
yours from your ratings. To train it, use *Rate for your taste* on the Label page
(about 60 photos at one place, mostly ones General or Yours would show first; rate six
places), then:

```bash
uv run calico eval taste      # would Yours beat General at places it never learned from, first picks too?
uv run calico train taste     # saves it only if it is shown to: clear of chance, at 5 places or more
uv run calico holdout draw    # once: set some ratings aside that no tuning sees (--holdout checks them)
```

Choose *Whose taste* on a search (`--template yours` on the command line), or set
`template = "yours"` in `calico.toml` to make it the default.

### Comparing first picks

To see whose first picks you prefer, use *Compare Strongest* on the Label page (or
`calico eval new-compare FOLDER`) once Yours is trained. Calico shows about 30 of
each template's strongest photographs at one place, merged and shuffled, without
saying whose. Rate each *Like*, *Not sure* or *Don't like*, five or six places, travel
among them, then:

```bash
uv run calico eval compare    # liked of rated among General's picks and Yours's
```

These ratings also train Yours (`calico train taste`): they are good-against-great
choices, as taste sessions now mostly are too.


### Rating a list of photos

To rate photos you (or a tool) chose, list their paths in a CSV file under a column
named `photo` and run:

```bash
uv run calico eval new-list FILE.csv   # a session of exactly those photos, shuffled
```

Rate them on the Label page as any taste session. The session knows only the photos, so a
list that mixes two versions' picks is rated without knowing whose each is. The ratings
train Yours.

### Rating stories

Stories hold back three frames of one moment (*variety*), and can be told *by roles*
(`-o tell_by=roles`): a close pair, an anchor with a person or animal in it or the widest
view, an echo or a change of scale, and never three shots of one space at one scale.
Today's way scores the mean of the three pairs and anchors on the strongest photos.
To see which way works for you, use *Rate stories* on the Label page (or `calico eval
new-stories FOLDER`): Calico tells six stories at one place each way, under each
template, today's way and by roles (up to 24; a story two ways told is shown once), and shows them shuffled, without saying which way told each. Mark each story *Works*, *Not sure* or *Doesn't*, and a photo that
alone does not belong *Doesn't fit*. To read one named set of options against today's
way instead, `calico eval new-stories FOLDER --variant NAME --template yours -o
KEY=VALUE` tells twelve stories each way under that template, reported as
`candidate:NAME`. Rate five or six places, then:

```bash
uv run calico eval stories    # liked over rejected, with no variety and each setting; how each way's stories were rated (today's, arranged, roles)
```

On a story result, 👍 means a photo fits the story and 👎 that it doesn't; `calico eval
stories` reports those beside the sessions, never pooled with them. Neither trains your
taste: they judge the story, not the photo.

To see whether stories you made yourself would be told, list each story's three
photos (left, middle, right, as catalogued) in a JSON file,
`{"stories": {"A": {"paths": [...]}}}`, and run:

```bash
uv run calico eval finished-stories stories.json --variant 'variety_weight=0'
```

Whether weights learned from your answers would rank stories better than today's
hand-set ones (each session judged by weights learned elsewhere; finished stories count
as stories that work):

```bash
uv run calico eval learned-stories stories.json
uv run calico eval learned-stories stories.json --swaps   # also learn from your stories with one photo swapped, as ones that don't work
uv run calico eval learned-stories --variant 'distinct_weight=0.3'   # today's way with other options, against today
```

A story turns on its middle photo, the anchor, and Calico's sense of a strong photo
is not yours of a strong anchor. *Could it anchor a story?* on the Label page (or
`calico eval new-anchors FOLDER`) shows about 60 photos from one place, each kind in
turn (aerial, wide, medium, close, macro; with or without a person or animal), from
strongest to weakest. Mark each *Could anchor*, *Not sure* or *Could not*. After five
or six places:

```bash
uv run calico eval anchors    # how well strength, a person or animal, a wide view find your anchors
```

### Your own vocabularies

Stories connect photos by `mood`, `place_type` and `surface`, Strongest filters by
`shot_kind`, and General judges with `aesthetic`. Each is a vocabulary: a TOML file of
labels, each with a few phrases CLIP compares photos with
(`src/calico/models/taxonomies/`). To change one for your own archive, copy it to
`<calico home>/taxonomies/` under the same name and edit it there; yours replaces the
built-in one, with no code. For example, a `mood.toml` that tells your kinds of light
apart:

```toml
description = "Light, mood and tone"
version = "mine-1"     # change it whenever you change the labels

[labels.fog]
prompts = ["a room filled with fog or haze", "light scattered by dust in the air"]

[labels.golden_light]
prompts = ["warm low sun across a floor", "golden evening light through windows"]

[labels.plain]
prompts = ["an evenly lit room", "flat, overcast light"]
```

Keep a catch-all label (`plain`) so photos with none of your labels are not spread
across them. A file under a new name (say `subject.toml`) loads, but no menu entry
reads it yet: only the names above change what Calico picks. Check a vocabulary by
eye before relying on it: run a story with that connection only (`-o
connections=mood`) and look at what it joins.

### Location names

A photo's location comes from its folder name, by each collection's rules.
Stray and doubled spaces never matter, and three-photo stories ignore capitals.
For the rest:

```bash
uv run calico locations          # every location, plus names that look like one place spelt twice
```

It suggests entries to paste into `calico.toml` after you have checked them:

```toml
[location_aliases]   # misspelt or renamed places -> the name to use
"Cobrett Building - Marlow, OH" = "Corbett Building - Marlow, OH"
```

To join names that differ only by a note in brackets, such as
"Mill (after the fire)", give the collection `--location-ignore '\([^)]*\)'`.

When the place is a folder *inside* the one at `--location-depth` (a city inside a
trip folder), add `--location-deepest`: the deepest folder that names a place is
the location, passing over folders that only organise, which you list with
`--not-place` (a regex matched against the whole name, any case):

```bash
uv run calico collections add travel "/Volumes/Archive/Travel" --location-depth 2 \
    --location-deepest --not-place 'Originals' --not-place 'Lightroom.*' \
    --location-ignore '\bDay\b.*$'   # "Calder Day 2 (5:8)" -> Calder
```

`Trip/Calder/Unedited/a.jpg` is then Calder, and `Trip/Originals/a.jpg` the trip.

### Screenshots

Phone screenshots and images saved from apps sit beside photos in many folders.
To leave PNG files with no camera recorded out of a collection's searches, add
`--exclude-screenshots` (or `exclude_screenshots = true` under the collection in
`calico.toml`); each search notes how many it left out. Leave it off for scans,
which are often camera-less PNGs too.

## Develop

```bash
make check   # Python + UI: format, lint, strict types, all tests + coverage — what CI runs
make fix
make ui-dev  # UI with hot reload (alongside `make serve`)
```

Rules for contributors (human or agent) are in [CLAUDE.md](CLAUDE.md).

Security problems: see [SECURITY.md](SECURITY.md).

## Third-party models

Calico downloads these weights the first time a feature needs them, and caches them on
your computer. Each has its own terms; the licences below are those stated on each
project's page (checked 20 November 2026).

| Model | Used for | Source | Licence |
|---|---|---|---|
| OpenCLIP ViT-B/16, LAION-2B (`laion2b_s34b_b88k`) | The main image and text embedding (`clip`) | [laion/CLIP-ViT-B-16-laion2B-s34B-b88K](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K) | MIT |
| SigLIP ViT-B/16 (WebLI) | Reading "Describe it" (`siglip`) | [timm/ViT-B-16-SigLIP](https://huggingface.co/timm/ViT-B-16-SigLIP) | Apache-2.0 |
| OpenAI CLIP ViT-L/14 | The larger judge and the input to LAION's score (`clip_large`) | [timm/vit_large_patch14_clip_224.openai](https://huggingface.co/timm/vit_large_patch14_clip_224.openai), from [openai/CLIP](https://github.com/openai/CLIP) | Apache-2.0 (weights, as tagged); MIT (OpenAI's code) |
| DINOv2 ViT-B/14 with registers (`lvd142m`) | "This exact place" embedding (`dinov2`) | [timm/vit_base_patch14_reg4_dinov2.lvd142m](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m), from [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache-2.0 |
| LAION improved aesthetic predictor (`sac+logos+ava1-l14-linearMSE.pth`) | An aesthetic score (`laion_aesthetic`) | [christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) | Apache-2.0 (the repository; no separate licence is stated for the weights) |

The model cards of both CLIP models describe them as research outputs and put "any
deployed use case" out of their intended scope; the LAION card also recommends its
training data for research use. Calico runs them on your own computer, for your own
photos. The libraries that load them are [open_clip](https://github.com/mlfoundations/open_clip)
(MIT) and [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0).

## Licence

Copyright (C) 2026 The Calico authors.

Calico is free software: you can redistribute it and/or modify it under the terms of the
GNU Affero General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version. It is distributed in the
hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](LICENSE).

The banner art was made by the project's owner, with AI, from photos of their cat. It is
shared under the same licence as the code.
