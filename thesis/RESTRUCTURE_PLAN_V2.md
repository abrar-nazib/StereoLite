# RESTRUCTURE PLAN V2 — new department format (2026-07-14 template)

Authority: `thesis/Thesis Writing Instructions/Resources/Thesis or Project Dissertation Format.pdf`
(4 pages, watermark "Red fonts are fixed for all reports"). Red headings = FIXED, keep verbatim.
Black headings = swappable with our own titles. This plan is the single coordination file for all
restructure agents. Branch: `thesis-v3-restructure`.

## Output layout

- New chapters: `thesis/book/mainmatter2/ch1.tex` .. `ch9.tex`
- New appendices: `thesis/book/appendices2/app_a.tex` .. `app_i.tex`
- OLD files in `mainmatter/`, `appendices/`, `frontmatter/` are READ-ONLY SOURCES. Never edit them.
- `Thesis.tex` gets updated only in the integration batch, after all chapter files exist.

## Shared agent contract (every agent follows ALL of this)

1. READ FIRST, in order:
   - This plan (your chapter's section below).
   - `/home/abrar/Research/stero_research_claude/.agents/skills/humanizer/SKILL.md` (writing style rules; the no em/en dash rule is HARD).
   - `/home/abrar/Research/stero_research_claude/.claude/skills/research-linguistics-expert/SKILL.md` (academic register for stereo literature).
   - Your assigned source files (listed per agent).
2. RED headings verbatim (capitalization normalized to Title Case consistently). BLACK headings may be renamed or replaced to fit our content.
3. JUDGE, don't just move: for every section you assemble, check the internal text actually matches the heading. If text belongs elsewhere per the ownership map, leave it to the owning agent (note it in your SOURCES comment). If a heading promises something the text doesn't deliver, rewrite the text (humanized) or retitle the black heading.
4. NUMBERS ARE FROZEN. No numeric value (EPE, ms, params, %, page counts, BDT/USD amounts, dataset sizes) may change. Copy them with their existing `\cite` / source citations. If a section genuinely needs a NEW number, take it only from `papers/verified_performance.md`, run `meta.json` files, or `papers/summaries/**`, and cite it. Never from memory.
5. LABELS AND ANCHORS. Never drop a `\label`. Move every figure/table/equation environment intact (graphic + caption + label) together with its prose. `\includegraphics` paths stay as-is (`figures/` is shared). Use `\ref`/`\eqref` only; never hard-code "Chapter 6" / "Section 3.2" / "Figure 4.1" numbers in prose. Keep existing `\cite` keys.
6. New diagrams: do NOT draw. Insert `% TODO-FABLE-DIAGRAM: <one-line description>` where one is needed and continue.
7. Prose register: first-person plural "we", past tense for what was done, present for what the system does. No `--` or `---` anywhere in prose. No bold-header bullet lists in body text. Captions: figure caption BELOW, table caption ABOVE.
8. Department = Mechatronics Engineering. Authors ordered Rahi (2008011) then Abrar (2008026). Supervisor Md Zunaid Hossen (Lecturer) is NOT an author.
9. Subsections OK (e.g. 8.2.1); sub-subsections (x.y.z.w) forbidden.
10. End every output file with a comment block:
    `% SOURCES: <old file:sections consumed>`
    `% DROPPED: <what was intentionally omitted and why>`
    `% MOVED-AWAY: <content seen but owned by another agent>`

## New structure with red/black classification

| New | Title (RED) | RED sections | BLACK slots (ours to fill) |
|---|---|---|---|
| Ch1 | Introduction | 1.1 Overview, 1.2 Background and Motivation, 1.3 Research Questions / Problem Statements, 1.4 Research Objectives, 1.5 Key Contributions, 1.6 Organization of the Report | none. NO Summary section |
| Ch2 | Literature Review | 2.1 Introduction, 2.2 Existing Research, 2.3 Comparative Analysis, 2.4 Summary | subsections under 2.2/2.3 free |
| Ch3 | Engineering Design and Methodology | 3.1 Introduction, 3.last Summary | all middle sections |
| Ch4 | System Development and Implementation | 4.1 Introduction, 4.last Summary | all middle sections (template suggests Hardware/Software Dev, Embedded Programming, AI Algorithm; pick what fits) |
| Ch5 | Experimental Setup and Data Collection | 5.1 Introduction, 5.last Summary | all middle sections (template suggests Setup, Test Bench, Instruments, Calibration, Procedure, Data Collection) |
| Ch6 | Results and Discussion | 6.1 Introduction, 6.last Summary | all middle sections (template suggests Experimental Results, Comparison, Error Analysis, Validation, Performance Evaluation, Discussion) |
| Ch7 | Socio-Economic Impact and Sustainability | ALL: 7.1 Introduction, 7.2 Impact of the Thesis on Societal, Health, Safety, Legal and Cultural Issues, 7.3 Impact of the Thesis on the Environment and Sustainability, 7.4 Summary | none |
| Ch8 | Thesis Management and Finance | ALL: 8.1 Introduction, 8.2 Management of the Thesis (with 8.2.1 Time Frame / Gantt Chart), 8.3 Overall Budget, 8.4 Resource Planning, 8.5 Individual and Team Work Contribution, 8.6 Summary | none |
| Ch9 | Conclusions, Limitations and Future Works | ALL: 9.1 Conclusion, 9.2 Limitations, 9.3 Future Works | none. NO Summary |
| App A | Addressing Knowledge Profile (KPs) in the Thesis | RED | |
| App B | Addressing Complex Engineering Problems | RED | |
| App C | Addressing Complex Engineering Activities | RED | |
| App D | Plagiarism Report | RED (placeholder page; actual Turnitin report inserted at submission) | |
| App E | AI Writing Report | RED (disclosure text + placeholder for supervisor-generated report) | |
| App F | List of Publications | optional, keep | |
| App G | Ethical Clearance | optional, keep short | |
| App H | Technical Specifications | extra (ours, kept) | |
| App I | PO and KPA Attainment Tracker | extra (ours, kept; pointers renumbered) | |

Frontmatter unchanged: Title page, Certificate (unnumbered) then Declaration (roman i),
Acknowledgement, Abstract, ToC, LoF, LoT, List of Abbreviations (nomenclature).

## Ownership map (old section -> new home)

| Old | New home | Owner agent |
|---|---|---|
| ch1 1.1, 1.2, 1.6 | ch1 (1.1, 1.2, 1.6) | ch1 |
| ch1 1.3 Objectives | ch1 1.4 Research Objectives | ch1 |
| ch1 1.4 Research Methodology | ch3 middle | ch3 |
| ch1 1.5 Project Planning + Gantt | ch8 8.2.1 | ch8 |
| ch1 1.7 Summary | dissolve (usable lines into 1.1/1.6) | ch1 |
| ch1 nomenclature `\nm{}` block | ch1 (keep intact) | ch1 |
| ch2 2.1, 2.2 | ch2 2.1, 2.2 Existing Research | ch2 |
| ch2 2.3 Requirements of Edge-Deployable Stereo | ch2 subsection under 2.2 (or feed ch3 requirements; coordinate via comments) | ch2 |
| ch2 2.4 Classification | ch2 subsection under 2.2 | ch2 |
| ch2 2.5 Cross-Domain Generalization and Benchmarks | ch2 2.3 Comparative Analysis (with capability matrix) | ch2 |
| ch2 2.6 Detailed Review of Adopted Methodology | ch3 middle | ch3 |
| ch2 2.7 Summary | ch2 2.4 | ch2 |
| ch3 3.1 | ch3 3.1 | ch3 |
| ch3 3.2 Problem Statements | ch1 1.3 (recast as research questions + problem statements) | ch1 |
| ch3 3.3-3.11 (requirements, criteria, steps, modeling, preliminary, detailed, assessment, summary) | ch3 middle + summary | ch3 |
| ch4 (socio-economic, 4 sections) | ch7 (all) | ch7 |
| ch5 5.2 CEP | App B | appx-kp |
| ch5 5.3 CEA | App C | appx-kp |
| ch5 K3/K4/K5/K8 knowledge-profile argument | App A (standalone) | appx-kp |
| ch5 5.1/5.4 intro/summary | dissolve into App A-C intros | appx-kp |
| ch6 6.1-6.3 | ch8 8.1-8.3 | ch8 |
| ch6 6.4 Component/Software Budget | ch8 subsection 8.3.x | ch8 |
| ch6 6.5 Summary | ch8 8.6 | ch8 |
| NEW 8.4 Resource Planning | write new (compute: RTX 3050, Modal credits, Kaggle T4, Jetson Orin Nano; data storage; human time split; source facts from old ch6 + app_a, numbers frozen) | ch8 |
| app_c 4. CRediT authorship table | ch8 8.5 | ch8 |
| app_c 5. Communication | ch8 8.2 (supervision meetings = management) | ch8 |
| ch7 7.2 Detailed Design Methodology | ch3 middle | ch3 |
| ch7 7.5 Review and Feedback | ch3 (design assessment) | ch3 |
| ch7 7.3 Selection of Parameters, Apparatus and Components | ch4 | ch4 |
| ch7 7.4 Optimization and Configuration Selection | ch4 | ch4 |
| ch7 7.7 Principle of Operation | ch4 | ch4 |
| ch7 7.6 Simulation and Experimental Setup | ch5 | ch5 |
| ch7 7.8 Testing and Measurement | ch5 | ch5 |
| ch7 7.1/7.9 | dissolve | ch4/ch5 |
| dataset descriptions (wherever they live in old ch2/ch7/ch8) | ch5 Data Collection sections; ch6 must not re-describe | ch5 |
| ch8 (results, 5 sections) | ch6 (all; may split under finer black headings) | ch6 |
| ch9 | ch9 (verify headings, no Summary) | ch9 |
| app_a Technical Specifications | App H | appx-tail |
| app_c 1. Ethics compliance | App G | appx-tail |
| app_c 2. Similarity report | App D | appx-tail |
| app_c 3. AI usage disclosure | App E | appx-tail |
| app_d PO/KPA tracker | App I (ALL location pointers renumbered to new structure) | appx-tail |
| app_e Publications | App F | appx-tail |

## Batches (max 5 parallel)

- B1: opus-ch1_intro, sonnet-ch2_litreview, opus-ch3_design, sonnet-ch4_implementation, sonnet-ch5_experimental
- B2: sonnet-ch6_results, haiku-ch7_socioimpact, sonnet-ch8_management, haiku-ch9_conclusions, sonnet-appx_kp_cep_cea
- B3: sonnet-appx_tail (D,E,F,G,H,I), haiku-frontmatter_audit, sonnet-master_integrate (Thesis.tex swap)
- B4: compile x3 + bibtex + nomencl; QA agents (format, dangling refs, number freeze diff); fable does drawio for every TODO-FABLE-DIAGRAM
- B5: fix batch as needed

Fable (main thread) inspects outputs after every batch before the next launches.

## QA checklist (B4)

- [ ] compile clean, 0 undefined refs / citations, ToC-LoF-LoT populated
- [ ] every RED heading present verbatim, in order
- [ ] grep prose for `--`, `---`, ` - ` misuse, hard-coded "Chapter [0-9]" / "Figure [0-9]" literals
- [ ] every `\label` of old book either present in new book or logged in DROPPED
- [ ] all `\includegraphics` targets exist; no figure orphaned (in old but not new, unlogged)
- [ ] numeric freeze: diff extracted numbers old vs new per chapter
- [ ] roman -> arabic transition at Ch1; captions below figures / above tables
- [ ] PO tracker (App I) pointers match new chapter numbers
