# Sources — `sources.yaml`, `src/sources.py`

Every document the system can search is listed in `sources.yaml` at the
project root: the textbook, public guidelines and Medicare policy documents,
your clinic's own protocols. `python3 -m src.ingest` indexes the entries with
`enabled: true` and nothing else.

Why a registry rather than "index whatever is in `data/raw/`": licensing. A
file on disk says nothing about whether it may be indexed, quoted to users,
or shipped inside `chunks.json`. An entry has to state its license, so adding
a source is a deliberate act with the terms written next to it.

```
 sources.yaml ──> load_registry() ──> enabled_sources()
                  validates fields     enabled entries, in file order;
                  (all problems        missing file = error with the
                   at once)            entry's url and notes
                                              |
                       +----------------------+----------------------+
                       |                                             |
              format: pdf_textbook                       pdf / html / nxml / docx / markdown
              ingest.build_chunks()                      loaders.load()  -> Units
              layout blocks + figures                    chunking.pack() -> chunks
              (unchanged since v1)                       (section-aware, overlapping)
                       |                                             |
                       +----------------------+----------------------+
                                              v
                              chunks.json + manifest.json  (ingest)
                                              v
                              vectors.npy + vector_keys.json  (embed,
                              only new or changed chunks are encoded)
```

## Status check

```bash
python3 -m src.ingest --list
```

One row per registry entry — whether it is enabled, whether its file is on
disk, its format and domain, how many chunks it has in the current index
(from `manifest.json`, `-` when not indexed), and its license — then a list
of problems with the command that fixes each:

```
index f972e1512122  built 2026-10-14T02:45:25+00:00  1491 chunks from 1 source(s)

id                                on   file  format        domain      chunks  license
mohs_layers                       yes  yes   pdf_textbook  clinical      1491  Copyrighted
statpearls_mohs                   no   no    html          clinical         -  CC BY-NC-ND 4.0
cms_ncci_ch3_2026                 no   yes   pdf           operations       -  US government work (public domain)
...

no problems
```

Problems it reports: an enabled entry whose file is missing, an enabled entry
that is not in the index yet, a disabled entry still in the index, an indexed
source that is no longer in `sources.yaml`, no `manifest.json` (an index
built before the registry), and `vectors.npy` not matching `chunks.json`.

## Adding a source

1. **Check the license first** (rules below). If it does not allow indexing
   and quoting, stop — do not add it "for later".
2. **Get the file** into `data/raw/<group>/` (gitignored). How depends on the
   publisher — see "Getting the files".
3. **Add an entry** to `sources.yaml`, with `enabled: false` at first:

   ```yaml
   - id: statpearls_mohs_periop          # lowercase, digits, underscores
     title: "Mohs Micrographic Surgery Safe and Effective Perioperative Care (StatPearls)"
     label: "StatPearls: Mohs perioperative care"   # shown in citations
     type: review                         # textbook|guideline|review|policy|protocol|article
     domain: clinical                     # clinical | operations
     format: html                         # picks the loader
     path: data/raw/statpearls/NBK611989.html
     url: https://www.ncbi.nlm.nih.gov/books/NBK611989/
     license: "CC BY-NC-ND 4.0"
     year: 2024
     enabled: false
     notes: "How to get it, caveats."
   ```

   `id` is stored on every chunk and is part of every chunk ID: choose it
   once and do not rename it (renaming re-embeds the whole source).
4. **Check it**: `python3 -m src.ingest --list` — the row should show
   `file yes`. A typo in a field name, a bad `format`/`domain`/`type`, or a
   duplicate `id` is reported here, all at once.
5. **Enable it** (`enabled: true`) and rebuild:

   ```bash
   python3 -m src.ingest && python3 -m src.embed
   ```

   `ingest` prints one line per source with its chunk counts by kind. A
   source with 0 chunks gets a WARNING — usually a wrong `format`, a scanned
   PDF with no text layer, or a web page saved before its article loaded.
6. **Look at what it produced** before trusting it: search for a fact you
   know is in the document (`python3 -m src.retrieve "..."`), and skim its
   chunks in `chunks.json` (`"source": "<id>"`) — headings in `section`,
   no navigation menus or reference lists posing as prose.
7. **Re-measure**: `python3 -m src.score --note "added <id>"`. A new source
   competes with the textbook for the top k; the gold labels in
   `questions.json` are textbook pages, so a drop in recall can mean the new
   source answered instead (read those answers), not that retrieval broke.
8. Once more than one source is indexed, set `CITATION_STYLE = "numbered"`
   in `src/config.py` — a page number alone no longer says which document.

To **remove** a source set `enabled: false` (keep the entry and its license
note) and re-run ingest + embed. Its chunks disappear from `chunks.json`;
the other sources' vectors are reused untouched.

## Licensing rules

- **Every entry states its license** in words; `sources.py` refuses an entry
  without one. Re-check the publisher's page before enabling — the notes in
  `sources.yaml` were checked on 2026-10-13 and are a starting point, not
  legal advice.
- **Never commit source files.** `data/raw/` is gitignored; keep every file
  there.
- **`chunks.json` contains the text of every indexed source**, and it is
  committed and deployed with the app (the HF Space reads it). Enabling a
  source therefore ships its text. Only enable in a deployed index what the
  license lets you redistribute verbatim to the app's audience.
- **CC BY-NC-ND 4.0** (StatPearls): non-commercial use, credit the authors,
  no altered versions. The index stores verbatim excerpts with their source
  label — that is quoting, which is fine; do not add summaries or rewrites
  of the text as indexed content.
- **US government works** (CMS manuals, LCDs, billing articles) are public
  domain, but the **CPT code descriptors inside them are AMA-copyrighted**.
  Coverage rules differ by Medicare contractor: index your contractor's LCD.
- **NCI PDQ**: text is free of copyright — credit the National Cancer
  Institute. Images are not reusable (the loaders keep text only), and
  excerpts may not be presented as an official PDQ summary.
- **Clinic protocols**: internal only. Never commit them, never index
  anything containing patient information, and do not enable them in an
  index that is deployed publicly.
- **Not listed on purpose**: NCCN Guidelines (the license bars using the
  content as input to any AI system) and the 2012 Mohs AUC document itself
  (copyrighted — the StatPearls AUC article covers it).

## Getting the files

`enabled_sources()` refuses to build while an enabled entry's file is
missing, and prints the entry's `url` and `notes` — the download
instructions live in `sources.yaml` next to each entry.

- **StatPearls (NCBI Bookshelf)** — save each page by hand: browser
  File > Save Page As, "Webpage, HTML only", into `data/raw/statpearls/`.
  Bookshelf forbids scripted bulk downloads from the website; the only
  allowed bulk route is the NLM LitArch FTP
  (`ftp.ncbi.nlm.nih.gov/pub/litarch/`, NBK ids in `file_list.csv`). If you
  get the `.nxml` files that way, set `format: nxml` and change the path's
  extension.
- **CMS NCCI manual** — the PDF link in the entry's `url`. CMS publishes a
  new edition every January: add it as a new entry (new `id` with the year,
  e.g. `cms_ncci_ch3_2027`), enable it, disable the old one.
- **LCD / billing article** — search the Medicare Coverage Database for
  "Mohs Micrographic Surgery", open your contractor's document, save the page
  as HTML.
- **NCI PDQ** — save the health-professional page as HTML. It is updated
  several times a year; re-save and re-run ingest + embed (only changed
  paragraphs are re-embedded).
- **Clinic documents** — `.docx` (or `.md` / `.txt` with `format: markdown`)
  in `data/raw/protocols/`. One entry per document.

## Formats

| format | read by | file | `page` | `locator` | `section` |
|---|---|---|---|---|---|
| `pdf_textbook` | `ingest.build_chunks()` — one chunk per layout block; figures extracted and linked to captions | the Mohs textbook PDF | page number | `"p. 188"` | `[]` |
| `pdf` | `loaders.load()` | any other PDF (CMS manuals) | page number | `"p. 12"` | heading path, if any |
| `html` | `loaders.load()` | a saved web page (StatPearls, LCDs, NCI PDQ) | `None` | section name | heading path |
| `nxml` | `loaders.load()` | JATS XML from the NLM LitArch FTP | `None` | section name | heading path |
| `docx` | `loaders.load()` | Word documents (clinic protocols) | `None` | section name | heading path |
| `markdown` | `loaders.load()` | `.md`, or plain `.txt` | `None` | section name | heading path |

`section` is the heading path, outermost first, e.g.
`["Indications", "Tumor characteristics"]`. How each loader finds the
article body, its headings and its tables is documented in `src/loaders.py`.

`pdf_textbook` is only for the Mohs textbook: its block-based path is kept
exactly as it was, so its chunk IDs, text and vectors — and every score in
`docs/experiments.md` — stay valid. Any other PDF uses `format: pdf`.

Every other format goes through `chunking.pack()` (see `src/chunking.py`):
chunks of about `CHUNK_TARGET_CHARS` (1000) characters that never cross a
section or a page, consecutive prose chunks overlapping by up to
`CHUNK_OVERLAP_CHARS` (150) characters of whole sentences, pieces under
`CHUNK_MIN_CHARS` (50) merged or dropped. Captions, references and tables
are chunks of their own, so retrieval can filter on `kind`. By default
(`RETRIEVE_KINDS` in `src/config.py`) prose, captions and tables are
searched — a staging table or a treatment-options table can be an answer —
and references are indexed but not searched.

Chunk IDs: `chunking.chunk_id(source_id, page, text)` for paginated sources;
for the rest the ID hashes the section path (the locator when there is
none) instead of the page, so the same paragraph under a same-named
subsection of two sections (NCI's "Cryosurgery" under both BCC and SCC) is
two chunks, and the loaders keep both. A reference repeated in two
sections' reference lists is kept once. The same text twice at the same
place in one source is a duplicate ID — ingest refuses and names the source
and locator.

### Loader options

`options:` in an entry passes settings to its loader; an unknown key is an
error, so a typo cannot silently do nothing.

```yaml
options:
  skip_tables: true          # drop tables and their captions. Tables are
                             # searched like prose, so long ones can crowd it
                             # out of the results - e.g. the ICD-10 code lists
                             # in a Medicare billing article
  exclude_sections:          # drop these headings' content, subsections included
    - Continuing Education Activity
    - Review Questions
```

`exclude_sections` **replaces** the defaults (StatPearls quiz and
learning-objective sections, the PDQ "About This PDQ Summary" boilerplate and
changelogs, tables of contents, acknowledgments, disclosures…), so
`exclude_sections: []` keeps everything. Matching ignores case, a leading
"1." / "F." and a trailing date or colon. The full list of defaults and the
reasons for each are in the `src/loaders.py` docstring. Page furniture (HTML
navigation, "Similar articles", …) is always dropped.

Options change the text that gets indexed, so changing one re-embeds that
source's affected chunks — and only those.

### Section headings in the embedding

With `EMBED_SECTION_HEADINGS = True` (`src/config.py`), a chunk with a
section path is embedded as

```
Indications > Tumor characteristics
<chunk text>
```

so a paragraph that only says "it is indicated when..." still matches a
question about Mohs indications. The textbook has no section path, so its
embedded text is the bare block text, exactly as before. The prefix is used
for the embedding only; `text` in `chunks.json` and the excerpts in the
prompt stay unchanged.

## Re-indexing

```bash
python3 -m src.ingest            # rebuild chunks.json + manifest.json from sources.yaml
python3 -m src.embed             # embed only new or changed chunks
python3 -m src.embed --full      # re-embed everything (after a library upgrade, or to be sure)
python3 -m src.ingest --list     # what is indexed, what is missing, what is stale
```

Run **both** after any change to `sources.yaml`, a source file, a loader, or
the chunking settings. `retrieve.py` checks that `vector_keys.json` names
exactly the chunks in `chunks.json`, in order, and refuses to search until
`embed` has been re-run.

`embed` keys every vector by `sha256(model | embedded text)`, so it re-uses
a vector whenever the same text was embedded before with the same model —
regardless of position — and prints `reused N, embedded M`. Things that
change the embedded text, and therefore re-embed the affected chunks:
editing or re-saving a document, a loader or chunking change, renaming a
heading, toggling `EMBED_SECTION_HEADINGS`. Changing `EMBEDDING_MODEL`
re-embeds everything. When nothing changed, `embed` does not even load the
model.

Commit `manifest.json` and `vector_keys.json` together with `chunks.json`
and `vectors.npy` — the app reads source labels from the manifest and
checks the key file.

See `docs/ingest.md` for the files and the first-upgrade path.
