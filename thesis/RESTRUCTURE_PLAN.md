# Thesis Restructure Plan: 5-chapter -> 9-chapter template

AUTHORITATIVE STRUCTURE: `thesis/Thesis Writing Instructions/Resources/thesis-or-project-template_1745300785.pdf`
(the earlier 5-chapter "Thesis Book Template.pdf" interpretation was WRONG).

Red headings in that template are FIXED for all reports; black ones adapt to the topic.
Every chapter opens with an Introduction section and closes with a Summary section
where the template shows them.

## Target chapter skeleton (headings verbatim where fixed)

- Chapter 1 Introduction
  - 1.1 Overview
  - 1.2 Background and Motivation
  - 1.3 Objectives
  - 1.4 Research Methodology
  - 1.5 Project Planning (1.5.1 Work Plan: Gantt chart)
  - 1.6 Organization of the Report
  - 1.7 Summary
- Chapter 2 Literature Review and Methodology
  - 2.1 Introduction
  - 2.2 Literature Review (subsections per era)
  - 2.3 Requirements of Edge-Deployable Stereo Matching
  - 2.4 Classification of Efficient Stereo Methods
  - 2.5 Cross-Domain Generalization and Benchmarks
  - 2.6 Detailed Review of the Adopted Methodology
  - 2.7 Summary
- Chapter 3 Design of StereoLite
  - 3.1 Introduction
  - 3.2 Problem Statements
  - 3.3 Design Requirements and Requisites
  - 3.4 Design Criteria
  - 3.5 Design Steps
  - 3.6 Modeling of Proposed System
  - 3.7 Modeling of Conceptual Design
  - 3.8 Preliminary Design
  - 3.9 Detailed Design (subsections: one per block)
  - 3.10 Design Assessment and Performance Indices
  - 3.11 Summary
- Chapter 4 Socio-Economic Impact and Sustainability
  - 4.1 Introduction
  - 4.2 Impact of the Thesis on Societal, Health, Safety, Legal and Cultural Issues
  - 4.3 Impact of the Thesis on the Environment and Sustainability
  - 4.4 Summary
- Chapter 5 Addressing Complex Engineering Problems and Activities
  - 5.1 Introduction
  - 5.2 Addressing Complex Engineering Problems
  - 5.3 Addressing Complex Engineering Activities
  - 5.4 Summary
- Chapter 6 Thesis Management and Finance
  - 6.1 Introduction
  - 6.2 Management of the Thesis (6.2.1 Time Frame)
  - 6.3 Overall Budget
  - 6.4 Component/Software Budget
  - 6.5 Summary
- Chapter 7 Design Methodology and Implementation
  - 7.1 Introduction
  - 7.2 Detailed Design Methodology
  - 7.3 Selection of Parameters, Apparatus and Components
  - 7.4 Optimization and Configuration Selection
  - 7.5 Review and Feedback
  - 7.6 Simulation and Experimental Setup
  - 7.7 Principle of Operation
  - 7.8 Testing and Measurement
  - 7.9 Summary
- Chapter 8 Results and Discussion
  - 8.1 Introduction
  - 8.2 Investigation and Analysis of Data
  - 8.3 Results (subsections per experiment)
  - 8.4 Discussion
  - 8.5 Summary
- Chapter 9 Conclusions, Limitations and Future Works
  - 9.1 Conclusion
  - 9.2 Limitations
  - 9.3 Future Works
- References
- Appendices (kept for department-mandated content not in the 9 chapters):
  - A Technical Specifications (trimmed)
  - B Ethics, Teamwork and Communication (old app_c, unchanged)
  - C PO and KPA Attainment Tracker (old app_d WITHOUT the CEP section, which moves to Ch5)
  - D List of Publications (old app_e)
  - (old app_b dissolves into Ch1 1.5 and Ch6)

## Source -> target map (each source section used EXACTLY ONCE)

Sources live in `thesis/book/mainmatter/ch{1..5}.tex` and `thesis/book/appendices/app_{a,b,d}.tex`.

| Target | Source |
|---|---|
| 1.1 Overview | NEW (condense old ch1 Background para 1 + Problem Statement essence) |
| 1.2 Background and Motivation | old ch1 "Background and Motivation" (rest) + "Problem Statement" (merged; keep fig edge_gap here) |
| 1.3 Objectives | old ch1 "Research Objectives and Scope" (objectives VERBATIM, unchanged) + "Rationale and Significance" first paragraph (why a new compact net) |
| 1.4 Research Methodology | NEW brief: design-by-controlled-ablation -> full training -> in-domain + zero-shot eval -> edge deployment; point to Ch3/7/8 |
| 1.5 Project Planning | old app_b "Project Timeline" (Gantt figure = 1.5.1 Work Plan) |
| 1.6 Organization of the Report | REWRITE for 9 chapters (bulleted) |
| 1.7 Summary | NEW 4-6 sentences |
| 2.1 Introduction | NEW short |
| 2.2 Literature Review | old ch2: Fundamentals, Deep Learning Transition, 3D Cost-Volume Era, Iterative Refinement, Foundation-Model Era (as subsections 2.2.x, content unchanged incl figs/eqs) |
| 2.3 Requirements of Edge-Deployable Stereo Matching | old ch3 "Problem Formulation and Design Requirements" (the requirements/envelope prose; NOT the four research questions, those go to 3.2) |
| 2.4 Classification of Efficient Stereo Methods | old ch2 "Efficient and Edge-Deployable Stereo" + its 5 subsections (incl taxonomy fig, HITNet + LiteAnyStereo panels, KD fig) |
| 2.5 Cross-Domain Generalization and Benchmarks | old ch2 "Cross-Domain Generalization" + "Benchmarks and Datasets" |
| 2.6 Detailed Review of the Adopted Methodology | old ch2 "Summary and Research Gap" reframed: the four inherited mechanisms (gwc cost, ConvGRU+convex upsample, GEV, tile planes) and the research gap |
| 2.7 Summary | NEW short |
| 3.1 Introduction | NEW short |
| 3.2 Problem Statements | old ch3 problem formulation: the four research questions + assumptions |
| 3.3 Design Requirements and Requisites | the hard envelope (< 3 M params, real time on 4-6 TOPS, < 200 MB inference memory, INT8-clean ops) stated as requirements list |
| 3.4 Design Criteria | the balance criteria: accuracy metrics AND latency AND memory jointly; no single-metric wins; cross-domain eval mandatory |
| 3.5 Design Steps | old ch3 "Design-Space Exploration Methodology" (100-pair harness protocol) |
| 3.6 Modeling of Proposed System | old ch3 "System Overview" (fig architecture + tab scales) |
| 3.7 Modeling of Conceptual Design | conceptual dataflow: the five stages + which literature mechanism each adapts (from System Overview second half) |
| 3.8 Preliminary Design | old ch3 "Summary of Design Decisions" (design-evidence audit table = the record of preliminary variants) |
| 3.9 Detailed Design | old ch3 block sections as subsections: Encoders, Tile Initialization, Recurrent Refinement, Plane Propagation, GEV + Fail-Soft Fusion, Plane Rendering + Convex Upsampling, From Disparity to 3D Reconstruction (all equations + figures unchanged) |
| 3.10 Design Assessment and Performance Indices | old ch3 "Evaluation Metrics" (EPE/bad-t/D1 equations) + param-budget figure |
| 3.11 Summary | NEW short |
| 4.1-4.4 | old ch4 "Societal, Environmental and Sustainability Aspects" SPLIT: societal/privacy/safety/assistive -> 4.2; environment/energy/training footprint -> 4.3. EXPAND both (health: eye-safe passive sensing vs laser; legal/cultural: on-device privacy, surveillance concerns; sustainability: commodity sensors, modest training footprint with numbers) |
| 5.2 CEP | old app_d "Complex Engineering Problem Attributes" + old ch1 "Rationale and Significance" CEP paragraph -> full P1-P7 attribute mapping table + K3/K4/K5/K8 prose |
| 5.3 CEA | NEW: A1-A5 activity mapping (range of resources; level of interaction; innovation; consequences to society/environment; familiarity) grounded in what the project actually did |
| 6.2 Management (6.2.1 Time Frame) | old app_b timeline phases as a table (reference the Gantt figure in 1.5.1, do NOT duplicate it) |
| 6.3 Overall Budget | old app_b "Expenses" table |
| 6.4 Component/Software Budget | old app_b "Economic Analysis" + split of cloud/compute vs hardware items |
| 7.2 Detailed Design Methodology | old ch3 "Training Objective" (loss equations) + "Datasets and Training Protocol" |
| 7.3 Selection of Parameters, Apparatus and Components | hyperparameter choices + hardware/software selection (draw from old ch3 "Implementation and Deployment Environment" + app_a environment section; app_a keeps the raw config table) |
| 7.4 Optimization and Configuration Selection | old ch3 "Optimization of the Network for Edge Inference" |
| 7.5 Review and Feedback | NEW short: ablation-gated iteration, EXPERIMENTS.md log discipline, supervisor reviews |
| 7.6 Simulation and Experimental Setup | old ch4 "Experimental Setup Summary" + eval drivers + Modal cloud + stereo rig description |
| 7.7 Principle of Operation | NEW: inference walkthrough (rectified pair in -> disparity out -> depth via Z=fB/d -> point cloud), referencing Ch3 blocks |
| 7.8 Testing and Measurement | measurement PROTOCOLS only (latency methodology, rectification sweep protocol, real-camera protocol, dual-axis eval) pulled from the protocol sentences of old ch4 sections; the RESULTS stay in Ch8 |
| 7.9 Summary | NEW short |
| 8.1 Introduction | NEW short (what is evaluated, one-paragraph setup recap) |
| 8.2 Investigation and Analysis of Data | old ch4 "Training Dynamics" + "Ablation Studies" (all 3 subsections, tables unchanged) |
| 8.3 Results | old ch4: SceneFlow Test, Middlebury 2014, Rectification, Efficiency and Deployment, Qualitative, 3D Reconstruction (as subsections; all tables/figures unchanged) |
| 8.4 Discussion | old ch4 "Discussion" |
| 8.5 Summary | NEW short |
| 9.1 Conclusion | old ch5 "Summary of Findings and Contributions" |
| 9.2 Limitations | old ch5 "Limitations" |
| 9.3 Future Works | old ch5 "Future Work" |

## Hard rules for every writer

1. Do NOT invent numbers. Move existing verified prose; new connective prose carries no new claims.
2. Keep ALL \label{...} names unchanged; each label appears exactly once in the whole book.
3. Keep equations byte-identical; LaTeX renumbers automatically.
4. NO en dash `--` or em dash `---` in prose. No \subsubsection.
5. Update hard-coded chapter references in moved prose: results discussion now Chapter 8; conclusions/future work Chapter 9; design Chapter 3; implementation/training Chapter 7; literature Chapter 2. Prefer \ref where a label exists.
6. Figure caption below, table caption above. \FloatBarrier before each \section is already the house style; keep it.
7. Objectives wording in 1.3 is FIXED verbatim (user requirement).
8. Every chapter needing Introduction/Summary sections gets them; keep each 4-8 sentences, no fluff.
9. Output files: `thesis/book/mainmatter/new/ch{N}.tex`, starting with `\chapter{...}` exactly as in the skeleton above.
