`/home/abrar/Research/stero_research_claude/.claude/agents/diagram-drawer.md`
[ Short description of what it does ]
Instruction prompt for the diagram-drawer agent.
[ Long description of what it does ]
This markdown file defines the persona, workflow, and rules for the diagram-drawer agent. It instructs the agent to produce diagrams (ASCII, Mermaid, or matplotlib) to communicate architectural or pipeline ideas clearly. It lays out the steps to pick the format, apply project conventions, ground the diagram in source code, and includes hard rules like "one diagram per call" and avoiding over-detailing.

`/home/abrar/Research/stero_research_claude/.claude/agents/stereo-vision-expert.md`
[ Short description of what it does ]
Instruction prompt for the stereo-vision-expert agent.
[ Long description of what it does ]
This file configures the stereo-vision-expert agent to act as an independent second eye for deep stereo matching design choices. It mandates a read-first, opine-second protocol where the agent must consult paper summaries, reference implementations, and equations before giving advice. The agent provides recommendations with cross-checkable citations and flags uncertainty, avoiding any code modification or unverified claims.

`/home/abrar/Research/stero_research_claude/.claude/skills/SKILL.md`
[ Short description of what it does ]
Defines the deep-research universal academic agent team.
[ Long description of what it does ]
This is the master skill file for a 13-agent deep research pipeline (v2.4). It outlines quick start commands, trigger conditions, orchestration workflows across 6 phases (Scoping, Investigation, Analysis, Composition, Review, Revision), and various operational modes like Socratic, Systematic Review, and Literature Monitoring. It serves as the central orchestrator for rigorous academic research tasks, providing routing and mode-selection logic.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/bibliography_agent.md`
[ Short description of what it does ]
Instruction prompt for the Bibliography Agent.
[ Long description of what it does ]
This file defines the Bibliography Agent responsible for systematic literature searches. It outlines the strategy framework involving defining parameters, executing searches, applying inclusion/exclusion criteria, and generating annotated bibliographies in APA 7.0 format. It also dictates PRISMA-style search documentation to ensure transparency and reproducibility.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/devils_advocate_agent.md`
[ Short description of what it does ]
Instruction prompt for the Devil's Advocate Agent.
[ Long description of what it does ]
This file configures an agent to act as a contrarian voice during research. It defines three mandatory checkpoints where the agent challenges assumptions, tests logical chains, detects logical fallacies (like confirmation bias or cherry-picking), and provides constructive criticism. It categorizes issues by severity to ensure arguments are robust and resilient to scrutiny.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/editor_in_chief_agent.md`
[ Short description of what it does ]
Instruction prompt for the Editor-in-Chief Agent.
[ Long description of what it does ]
This file sets up the Editor-in-Chief Agent to review research reports with Q1 journal rigor. It evaluates submissions across five dimensions: Originality & Contribution, Methodological Rigor, Evidence Sufficiency, Argument Coherence, and Writing Quality. The agent assigns scores and delivers a final verdict (Accept, Minor/Major Revision, or Reject) along with specific, actionable feedback.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/ethics_review_agent.md`
[ Short description of what it does ]
Instruction prompt for the Ethics Review Agent.
[ Long description of what it does ]
This agent prompt focuses on research integrity and AI ethics. It mandates checks for AI disclosure, attribution integrity, dual-use screening, fair representation, and human subjects ethics (IRB). The agent verifies citations against Retraction Watch, checks for self-citation bias, and can halt the delivery of a research report if critical ethical violations are detected.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/meta_analysis_agent.md`
[ Short description of what it does ]
Instruction prompt for the Meta-Analysis Agent.
[ Long description of what it does ]
This file defines the procedures for the Meta-Analysis Agent to conduct quantitative evidence synthesis. It includes decision flowcharts for determining whether to pool data, methods for calculating effect sizes (continuous, binary, time-to-event), assessing heterogeneity (I²), and generating forest plot data. It also covers narrative synthesis frameworks and GRADE certainty of evidence assessments.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/monitoring_agent.md`
[ Short description of what it does ]
Instruction prompt for the Monitoring Agent.
[ Long description of what it does ]
This file configures an agent to generate post-research literature monitoring digests and alerts. It uses completed bibliographies to track new publications, retractions, contradictory findings, and author activities. The agent outputs actionable markdown digest templates and alert configurations for the user to set up on external platforms like Google Scholar or PubMed.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/report_compiler_agent.md`
[ Short description of what it does ]
Instruction prompt for the Report Compiler Agent.
[ Long description of what it does ]
This prompt guides the Report Compiler Agent in drafting and polishing academic reports in APA 7.0 format. It details structural outlines for both full and quick modes, covering abstracts, methodology, findings, discussions, and appendices. It also incorporates a writing quality check, style calibration, and a mandatory AI disclosure statement to maintain high academic writing standards.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/research_architect_agent.md`
[ Short description of what it does ]
Instruction prompt for the Research Architect Agent.
[ Long description of what it does ]
This agent is tasked with designing methodological blueprints. The file outlines decision trees for selecting research paradigms, methods, data strategies, analytical frameworks, and validity criteria based on the research question. It also enforces the inclusion of IRB planning for human subjects and standard reporting guidelines like PRISMA or CONSORT.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/research_question_agent.md`
[ Short description of what it does ]
Instruction prompt for the Research Question Agent.
[ Long description of what it does ]
This file defines the agent responsible for transforming vague topics into precise research questions using the FINER framework (Feasible, Interesting, Novel, Ethical, Relevant). It outlines steps for topic decomposition, question generation, scope definition, and handling Socratic mode guidance where the user is led to derive the research question themselves.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/risk_of_bias_agent.md`
[ Short description of what it does ]
Instruction prompt for the Risk of Bias Agent.
[ Long description of what it does ]
This document defines an agent that evaluates the risk of bias in systematic review studies. It utilizes the Cochrane RoB 2 tool for randomized trials and ROBINS-I for non-randomized studies. The agent sequentially answers signaling questions to derive domain-level judgments, ultimately producing a traffic-light visualization table representing the overall bias risk of each study.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/socratic_mentor_agent.md`
[ Short description of what it does ]
Instruction prompt for the Socratic Mentor Agent.
[ Long description of what it does ]
This file configures an agent that guides users through clarifying their research via Socratic questioning over a 5-layer model (Problem Framing, Methodology, Evidence, Critical Self-Review, Significance). It tracks convergence signals and extracts 'INSIGHT' tags to compile a final research plan, intentionally withholding direct answers to promote independent critical thinking.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/source_verification_agent.md`
[ Short description of what it does ]
Instruction prompt for the Source Verification Agent.
[ Long description of what it does ]
This file defines an agent that acts as a quality gatekeeper for evidence. It grades sources on a 7-level evidence hierarchy, performs publication venue and author credibility checks, and actively hunts for predatory journals, hallucinated references, and conflicts of interest. It uses automated DOI and web search spot-checks to ensure no fabricated citations enter the pipeline.

`/home/abrar/Research/stero_research_claude/.claude/skills/agents/synthesis_agent.md`
[ Short description of what it does ]
Instruction prompt for the Synthesis Agent.
[ Long description of what it does ]
This prompt instructs the Synthesis Agent on how to integrate findings across multiple sources rather than summarizing them sequentially. It outlines thematic and narrative synthesis methods, strategies for mapping evidence convergence and divergence, contradiction resolution, and gap analysis, producing a comprehensive synthesis report that bridges raw data and theoretical frameworks.

`/home/abrar/Research/stero_research_claude/.claude/skills/diagram-drawer/SKILL.md`
[ Short description of what it does ]
High-level description of the diagram-drawer skill.
[ Long description of what it does ]
This is the root configuration file for the diagram-drawer skill. It describes the purpose of the skill, the supported output formats (ASCII, Mermaid, matplotlib), the constituent files like `conventions.md` and examples, and provides a quick decision tree for when to use each format. It sets hard rules for the drawer to ensure visual consistency and clarity.

`/home/abrar/Research/stero_research_claude/.claude/skills/diagram-drawer/ascii_examples.md`
[ Short description of what it does ]
Examples of ASCII diagrams.
[ Long description of what it does ]
This markdown file provides reference patterns for drawing ASCII diagrams inline in chat. It contains 8 distinct examples, including linear pipelines, block dictionaries, side-by-side comparisons, dataflow trees, and knob-effect annotations. These patterns serve as templates for the diagram-drawer agent to copy and adapt for low-friction, immediate visual explanations.

`/home/abrar/Research/stero_research_claude/.claude/skills/diagram-drawer/conventions.md`
[ Short description of what it does ]
Visual conventions for drawing diagrams.
[ Long description of what it does ]
This document establishes the consistent visual vocabulary for all diagrams in the project. It specifies the color palette (e.g., pale blue for encoders, warm yellow for cost volumes) and glyph vocabulary (e.g., prisms for cost volumes, red dots for supervision). It also lists annotation patterns and anti-patterns to avoid, ensuring uniformity across matplotlib, Mermaid, and ASCII outputs.

`/home/abrar/Research/stero_research_claude/.claude/skills/diagram-drawer/helpers/diag_helpers.py`
[ Short description of what it does ]
Python helpers for matplotlib architecture diagrams.
[ Long description of what it does ]
This Python module provides reusable functions to draw standardized matplotlib elements. It defines the project's color palette as constants and offers helper functions like `box`, `txt`, `arrow`, `sup_dot`, `cv_prism`, and `loop_glyph`. This ensures that any new script generating a matplotlib architecture diagram uses identical styling to existing project figures.


`/home/abrar/Research/stero_research_claude/.claude/skills/diagram-drawer/matplotlib_examples.md`
[ Short description of what it does ]
Examples and templates for matplotlib architecture diagrams.
[ Long description of what it does ]
This file provides reference code and guidelines for using matplotlib to generate publication-quality PNG/PDF architecture diagrams. It explains when to use matplotlib (for saved figures, dense architectures with prisms, presentations) versus Mermaid or ASCII. It includes templates for minimal pipelines and block-level drilldowns, and establishes standards like saving ad-hoc diagrams to `/tmp/`.

`/home/abrar/Research/stero_research_claude/.claude/skills/diagram-drawer/mermaid_examples.md`
[ Short description of what it does ]
Examples and templates for Mermaid flowchart diagrams.
[ Long description of what it does ]
This document contains Mermaid diagram templates designed for clear, inline IDE rendering. It provides patterns for linear pipelines, branched dataflows, side-by-side architecture comparisons, block-internal drilldowns, knob-effect grids, and ablation decision trees. It includes a required CSS-like class definitions block to maintain the project's consistent color palette across all diagrams.

`/home/abrar/Research/stero_research_claude/.claude/skills/examples/exploratory_research.md`
[ Short description of what it does ]
Example of an exploratory research workflow.
[ Long description of what it does ]
This file demonstrates a full end-to-end execution of the deep-research agent pipeline on an exploratory topic (AI in higher ed quality assurance). It shows the outputs at each phase: scoping (FINER assessment), investigation (PRISMA flow), analysis (synthesis of themes and gaps), composition, and review (including ethics and devil's advocate checks), culminating in an APA 7.0 report.

`/home/abrar/Research/stero_research_claude/.claude/skills/examples/fact_check_mode.md`
[ Short description of what it does ]
Example of the fact-check operational mode.
[ Long description of what it does ]
This example illustrates how the `source_verification_agent` processes a list of specific claims (using Taiwan higher education as the test domain). It breaks down the verification of 7 distinct claims, assigning verdicts like "Verified", "Warning — Partially True", "Unverifiable", or "False", and providing correction suggestions based on authoritative sources.

`/home/abrar/Research/stero_research_claude/.claude/skills/examples/handoff_to_paper.md`
[ Short description of what it does ]
Example of transitioning from deep-research to academic-paper writing.
[ Long description of what it does ]
This document details the handoff process where completed outputs from the deep-research skill (RQ brief, methodology blueprint, annotated bibliography, synthesis) are ingested by the `academic-paper` agents. It shows how the intake agent skips early stages and maps research artifacts directly into an accelerated writing pipeline to draft the final manuscript.

`/home/abrar/Research/stero_research_claude/.claude/skills/examples/policy_analysis.md`
[ Short description of what it does ]
Example of a comparative policy analysis workflow.
[ Long description of what it does ]
This file presents a full-mode pipeline execution focused on policy analysis (performance-based funding in OECD higher education). It highlights the use of comparative matrix synthesis, devil's advocate challenges on case selection and effect causality (e.g., Hawthorne effect), and editor checks on evidence strength, resulting in a robust, multi-country policy report.

`/home/abrar/Research/stero_research_claude/.claude/skills/examples/review_mode.md`
[ Short description of what it does ]
Example of the review mode for policy recommendations.
[ Long description of what it does ]
This example demonstrates the standalone `review` mode involving the editor-in-chief, ethics, and devil's advocate agents. It shows how a short, user-provided policy recommendation summary is critiqued for data accuracy, logical leaps, missing stakeholder perspectives, and feasibility issues, producing a consolidated revision recommendation summary.

`/home/abrar/Research/stero_research_claude/.claude/skills/examples/socratic_guided_research.md`
[ Short description of what it does ]
Example of the Socratic guided research mode.
[ Long description of what it does ]
This document captures a 12-round dialogue between a user and the `socratic_mentor_agent`. It illustrates the 5-layer conversational framework (Problem Framing, Methodology Reflection, Evidence Design, Critical Self-Examination, Significance & Contribution) used to transform a vague interest into a precise, methodologically sound research plan with explicitly extracted INSIGHT tags.

`/home/abrar/Research/stero_research_claude/.claude/skills/examples/systematic_review.md`
[ Short description of what it does ]
Example of the systematic literature review (lit-review) mode.
[ Long description of what it does ]
This file showcases a targeted `lit-review` execution using only the bibliography, source verification, and synthesis agents. It details a PRISMA-compliant search flow for micro-credentials, a source quality matrix that flags predatory journals and COIs, and a condensed synthesis narrative highlighting strong evidence, contested themes, and knowledge gaps.

`/home/abrar/Research/stero_research_claude/.claude/skills/open3d-expert/SKILL.md`
[ Short description of what it does ]
Defines the open3d-expert agent skill.
[ Long description of what it does ]
This is the main skill instruction file for the Open3D expert agent. It provides a mental model of Open3D (Geometry, Pipelines, Visualization), a function selection guide, and specific recipes for common tasks in the project like cleaning point clouds, meshing (Ball-pivoting, Poisson), headless rendering/animation, ICP registration, and TSDF integration.

`/home/abrar/Research/stero_research_claude/.claude/skills/open3d-expert/references/mesh_ops.md`
[ Short description of what it does ]
Reference guide for Open3D TriangleMesh operations.
[ Long description of what it does ]
This reference details how to construct, process, and manipulate triangle meshes in Open3D. It covers surface reconstruction techniques (Ball pivoting, Poisson, Alpha shapes), mesh cleaning operations, smoothing algorithms (Taubin vs Laplacian), decimation, point sampling (Poisson disk), and geometric transformations, complete with code snippets.

`/home/abrar/Research/stero_research_claude/.claude/skills/open3d-expert/references/point_cloud_ops.md`
[ Short description of what it does ]
Reference guide for Open3D PointCloud operations.
[ Long description of what it does ]
This document provides recipes for working with point clouds in Open3D. It includes instructions for reading/writing various formats, voxel and uniform downsampling, statistical and radius outlier removal, normal estimation and orientation, spatial cropping, KD-tree neighborhood searches, and computing distances (e.g., Chamfer distance).

`/home/abrar/Research/stero_research_claude/.claude/skills/open3d-expert/references/reconstruction_pipeline.md`
[ Short description of what it does ]
Guide to the Open3D RGBD reconstruction system pipeline.
[ Long description of what it does ]
This file outlines the four-stage Open3D reconstruction pipeline: make_fragments, register_fragments, refine_registration, and integrate_scene. It explains how to process long RGBD sequences into a globally aligned dense 3D mesh, includes instructions for capturing custom datasets with RealSense, and discusses color map optimization and practical voxel sizes.

`/home/abrar/Research/stero_research_claude/.claude/skills/open3d-expert/references/rgbd_pipeline.md`
[ Short description of what it does ]
Reference for handling RGBD images and odometry in Open3D.
[ Long description of what it does ]
This document explains how to construct and use `RGBDImage` objects from separate color and depth images. It covers camera intrinsic definitions, converting RGBD to PointClouds, a specific shortcut for turning stereo disparity maps into depth/RGBD, and computing odometry between consecutive RGBD frames.

`/home/abrar/Research/stero_research_claude/.claude/skills/open3d-expert/references/visualization.md`
[ Short description of what it does ]
Guide to Open3D visualization and headless rendering.
[ Long description of what it does ]
This reference details methods for visualizing 3D data in Open3D. It covers simple one-shot `draw_geometries`, configuring the `Visualizer` class for programmatic interaction and capturing frames, animation callbacks, and setting up the modern `OffscreenRenderer` (Filament-based) for true headless rendering of smooth rotating animations.

`/home/abrar/Research/stero_research_claude/.claude/skills/references/apa7_style_guide.md`
[ Short description of what it does ]
Quick reference for APA 7th Edition formatting.
[ Long description of what it does ]
This guide provides the formatting rules required by the report compiler and editor agents. It details document layout, the 5-level heading structure, rules for in-text citations (parenthetical, narrative, direct quotes), and reference list formats for various source types (journals, books, webpages, datasets). It also lists common formatting errors to avoid.

`/home/abrar/Research/stero_research_claude/.claude/skills/references/equator_reporting_guidelines.md`
[ Short description of what it does ]
Mapping of research designs to EQUATOR reporting guidelines.
[ Long description of what it does ]
This document maps various research designs (RCTs, systematic reviews, observational, qualitative) to their respective EQUATOR Network reporting standards (PRISMA, CONSORT, STROBE, COREQ, SQUIRE). It provides condensed checklists for these major guidelines to help the research architect select appropriate standards and the report compiler ensure completeness.

`/home/abrar/Research/stero_research_claude/.claude/skills/references/ethics_checklist.md`
[ Short description of what it does ]
Comprehensive ethics checklist for AI-assisted research.
[ Long description of what it does ]
This file serves as the strict guideline for the `ethics_review_agent`. It mandates checks across 8 areas: AI disclosure statements, attribution integrity (preventing hallucinated citations), dual-use assessment, fair representation (bias and sensitive topics), data source ethics, conflicts of interest, reproducibility, and human subjects ethics (IRB protocols).

`/home/abrar/Research/stero_research_claude/.claude/skills/references/failure_paths.md`
[ Short description of what it does ]
Map of failure scenarios and recovery strategies in the research pipeline.
[ Long description of what it does ]
This document catalogs 12 distinct failure paths (F1-F12) across all agent modes, ranging from non-converging RQs and insufficient literature to critical ethical blocks and interdisciplinary bridging failures. For each scenario, it defines trigger conditions, user notification templates, handling steps, and specific recovery paths to prevent workflow dead-ends.

`/home/abrar/Research/stero_research_claude/.claude/skills/references/interdisciplinary_bridges.md`
[ Short description of what it does ]
Guide to finding and applying cross-disciplinary connections.
[ Long description of what it does ]
This reference aids the synthesis and architect agents in identifying connections across academic disciplines. It details patterns like "Shared Concept, Different Names," "Methodological Transfer," and "Problem Reframing" (e.g., viewing student dropout through education, economics, and sociology lenses). It provides a practical guide for researchers to borrow concepts and avoid shallow interdisciplinarity.


`file:///home/abrar/Research/stero_research_claude/.claude/skills/references/irb_decision_tree.md`
[ Decision tree for IRB/Ethics requirements ]
[ A comprehensive guide and decision tree flowchart for determining when Institutional Review Board (IRB) or ethics approval is required for a research project. Covers human subjects, data privacy, exemptions, and outlines the documentation process for both clinical and non-clinical research. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/references/literature_monitoring_strategies.md`
[ Automated literature search techniques ]
[ Defines automated and manual strategies for monitoring academic literature. Covers RSS feeds, citation alerts, query formulation (PubMed, arXiv), and methods for maintaining up-to-date knowledge in rapidly evolving research fields. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/references/logical_fallacies.md`
[ List of logical fallacies to avoid ]
[ A dictionary of 15 common logical fallacies (e.g., ad hominem, straw man, confirmation bias, post hoc) to help agents and researchers identify and avoid flawed reasoning during literature review, argumentation, and hypothesis formulation. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/references/methodology_patterns.md`
[ Study design patterns and methodologies ]
[ A catalog of common research study designs, including Randomized Controlled Trials (RCT), cohort studies, case-control studies, and observational studies. Explains the strengths, weaknesses, and appropriate use cases for each methodology. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/references/mode_selection_guide.md`
[ Guide for agent research modes ]
[ Describes the various operational modes supported by the agentic system (e.g., `full`, `quick`, `review`, `lit-review`, `fact-check`, `socratic`, `systematic-review`) and provides guidelines on when to select each mode based on the user's research needs. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/references/preregistration_guide.md`
[ Guidelines for preregistering studies ]
[ Provides best practices and templates for preregistering research protocols on platforms like OSF or PROSPERO. Aims to reduce publication bias and p-hacking by enforcing a priori documentation of hypotheses, methodologies, and analysis plans. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/references/socratic_questioning_framework.md`
[ Questioning framework for Socratic mentor ]
[ Outlines a 6-type Socratic questioning framework used by the `socratic_mentor` agent to guide users through problem-solving and critical thinking. Includes question types for clarifying concepts, probing assumptions, and examining consequences. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/references/source_quality_hierarchy.md`
[ 7-level evidence hierarchy and grading rubric ]
[ Defines a 7-level pyramid of evidence quality (from expert opinion up to systematic reviews/meta-analyses) and provides a grading rubric for agents to critically assess and weight the reliability of different scientific sources. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/references/systematic_review_toolkit.md`
[ Guidelines for systematic reviews ]
[ A comprehensive toolkit detailing the protocols and standards for conducting systematic reviews. Includes guidance on PRISMA guidelines, Cochrane methodologies, and tools for assessing Risk of Bias (RoB 2, ROBINS-I) and certainty of evidence (GRADE). ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/SKILL.md`
[ Entry point for stereo-vision-expert ]
[ Defines the `stereo-vision-expert` skill, establishing it as the authoritative source for stereo vision architectures (specifically the RAFT/IGEV lineage). Enforces a strict "read-first, opine-second" rule using `papers_index.md` before forming design opinions. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/architectures.md`
[ Canonical stereo model architectures ]
[ Top-down end-to-end descriptions of the most-referenced stereo matching models. Covers models like RAFT-Stereo, IGEV-Stereo, PSMNet, BGNet, HITNet, FoundationStereo, DEFOM-Stereo, and Pip-Stereo, detailing their parameter counts, latency, and structure. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/blocks.md`
[ Per-layer/block primitives for stereo models ]
[ An authoritative reference dictionary for individual network blocks (e.g., encoders, cost volumes, aggregators, upsamplers) used across the stereo corpus. Analyzes parameters, costs, hyperparameters, and pros/cons of each block primitive. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/design_lessons.md`
[ Cross-corpus design lessons for stereo models ]
[ Synthesizes high-level, cross-corpus design patterns and lessons learned from stereo matching literature. Details findings such as the necessity of iterative refinement, the dominance of warm starts, aggregation resolution choices, and metric evaluation pitfalls. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/equations.md`
[ Canonical equations for stereo networks ]
[ A cheat sheet of essential mathematical formulations and equations used in stereo matching. Includes stereo geometry, global energy formulation, SGM penalties, correlation volumes, soft-argmin, ConvGRU updates, upsampling masks, and loss functions like sequence loss and D1 metrics. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/papers_index.md`
[ Map from papers to raw PDFs and summaries ]
[ A comprehensive index mapping stereo vision research papers to their raw PDF file paths and their summarized markdown files. Categorizes papers into tiers and themes such as Foundation Models, Efficient/Edge, Iterative, End-to-End, Transformers, NAS, and Datasets. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/README.md`
[ Index of reference implementations ]
[ Documentation detailing the available reference implementations for iterative stereo models. Explains when to consult the code over paper summaries and provides a directory map for RAFT-Stereo, IGEV-Stereo, Selective-Stereo, CoEx, and LiteAnyStereo source files. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/igev_stereo/extractor.py`
[ IGEV-Stereo extraction networks ]
[ PyTorch implementation of the feature and context extraction networks used in IGEV-Stereo. Includes `BasicEncoder` and `MultiBasicEncoder` classes, defining the convolutional layers, residual blocks, and MobileNetV2-based `Feature` extractor. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/igev_stereo/geometry.py`
[ Combined Geometry Encoding Volume for IGEV-Stereo ]
[ PyTorch implementation of the `Combined_Geo_Encoding_Volume` class for IGEV-Stereo. This builds and manages the geometric cost volume pyramid and provides methods for differentiable lookups based on disparity and coordinates. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/igev_stereo/igev_stereo.py`
[ Main definition of IGEV-Stereo network ]
[ The primary PyTorch model file defining the `IGEVStereo` class. Orchestrates the full forward pass, including feature extraction, Combined Geometry Encoding Volume construction, warm-start initialization, the multi-level ConvGRU iterative refinement loop, and upsampling. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/igev_stereo/submodule.py`
[ Helper modules for IGEV-Stereo ]
[ Defines utility neural network modules for IGEV-Stereo, including basic convolutional blocks, group-wise correlation volume construction, feature attention layers, and disparity regression. Contains foundational building blocks used across the network's architecture. ]


`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/igev_stereo/update.py`
[ Iterative update blocks for IGEV-Stereo ]
[ Defines the `BasicMultiUpdateBlock`, `ConvGRU`, `SepConvGRU`, `FlowHead`, `DispHead`, and `BasicMotionEncoder`. These modules form the recurrent update machinery for IGEV-Stereo, refining the disparity predictions over multiple iterations at different scales (e.g., 1/4, 1/8, 1/16). Includes convex upsampling mask prediction. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/lite_any_stereo/aggregation.py`
[ 2D Aggregation network for LiteAnyStereo ]
[ PyTorch implementation of `Aggregation2D`, utilizing ConvNeXt blocks, LayerNorm, and an `AttentionModule2D` to process and aggregate cost volumes efficiently. Designed to reduce computational overhead while maintaining spatial context through left-image attention injection. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/lite_any_stereo/fnet.py`
[ Feature Network for LiteAnyStereo ]
[ Implements `FeatureNet` using a pretrained MobileNetV2 backbone from `timm`. Extracts multi-scale features and processes them through FPN (Feature Pyramid Network) layers to generate robust hierarchical representations for cost volume construction. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/lite_any_stereo/liteanystereo.py`
[ Main LiteAnyStereo network architecture ]
[ Defines the `LiteAnyStereo` class, orchestrating feature extraction (`FeatureNet`), 3D/2D cost aggregation, softmax-based disparity regression, and refinement upsampling. Provides a highly efficient, lightweight architecture suitable for real-time stereo matching. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/lite_any_stereo/profile_speed.py`
[ Inference speed benchmarking script ]
[ A utility script to profile the latency and throughput of the LiteAnyStereo model. Measures mean/median milliseconds and FPS using dummy tensors, providing options for AMP (Automatic Mixed Precision) and input padding. Useful for validating real-time performance claims. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/lite_any_stereo/submodule.py`
[ Submodules and utilities for LiteAnyStereo ]
[ Contains foundational building blocks such as `BasicConv2d`, `BasicDeconv2d`, `FPNLayer`, group-wise correlation functions, spatial transformers, and context upsampling functions used extensively throughout the LiteAnyStereo architecture. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/raft_stereo/corr.py`
[ Correlation volume implementation for RAFT-Stereo ]
[ Implements various correlation volume methods (e.g., `CorrBlock1D`, `CorrBlockFast1D`, `PytorchAlternateCorrBlock1D`). Constructs the all-pairs correlation pyramid and handles differentiable sampling based on disparity/flow coordinates. Critical for the RAFT matching process. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/raft_stereo/extractor.py`
[ Feature and context extractors for RAFT-Stereo ]
[ Defines `BasicEncoder` and `MultiBasicEncoder` using residual and bottleneck blocks. These networks are responsible for processing the input images to produce multi-scale feature maps for matching and context features for the recurrent update loop. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/raft_stereo/raft_stereo.py`
[ Main RAFT-Stereo network definition ]
[ The primary `RAFTStereo` class encapsulating the full architecture. Coordinates the feature extraction, correlation pyramid construction, iterative multi-level GRU updates, and flow upsampling. Implements the standard optical-flow-adapted-for-stereo paradigm. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/raft_stereo/update.py`
[ Iterative update loop for RAFT-Stereo ]
[ Contains the `BasicMultiUpdateBlock`, `ConvGRU`, and `BasicMotionEncoder`. Processes the correlation samples, flow, and context features to iteratively predict $\Delta$ flow/disparity, updating the hidden state and predicting upsampling weights at each step. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/selective_igev/extractor.py`
[ Enhanced extractors for Selective-IGEV ]
[ Implements `BasicEncoder`, `MultiBasicEncoder`, and a `Feature` class using MobileNetV2 from `timm`. Tailored for Selective-IGEV to extract hierarchical features while balancing performance and efficiency, employing various normalization techniques (Batch, Instance, Group). ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/selective_igev/geometry.py`
[ Geometry Encoding Volume for Selective-IGEV ]
[ Implements the `Combined_Geo_Encoding_Volume`, combining initial correlations with a geometry-aware volume. Provides methods to sample from these combined volumes during the iterative refinement phase, integrating structural constraints into the updates. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/selective_igev/igev_stereo.py`
[ Main Selective-IGEV network definition ]
[ Defines the `IGEVStereo` class adapted for Selective-IGEV. Incorporates Spatial and Channel Attention modules (`sam`, `cam`), builds the Geometry Encoding Volume, computes the initial disparity, and refines it through `BasicSelectiveMultiUpdateBlock` across iterations. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/selective_igev/submodule.py`
[ Shared submodules for Selective-IGEV ]
[ Contains convolution blocks, group-wise correlation functions, feature attention layers (`FeatureAtt`), and context upsampling routines needed to construct the multi-scale, attention-driven Selective-IGEV architecture. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/selective_igev/update.py`
[ Selective GRU update blocks for Selective-IGEV ]
[ Implements `SelectiveConvGRU` alongside standard GRUs. Uses spatial and channel attention signals to selectively route information through small ($1 \times 1$) or large ($3 \times 3$) kernel GRUs, drastically reducing convergence time and computational cost per iteration. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/selective_raft/corr.py`
[ Correlation blocks for Selective-RAFT ]
[ Implements the correlation pyramid and sampler (e.g., `CorrBlock1D`, `AlternateCorrBlock`) for the Selective-RAFT variant. Reuses core RAFT correlation concepts but adapts them for integration with selective recurrent units. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/selective_raft/extractor.py`
[ Extractor networks for Selective-RAFT ]
[ Defines the `BasicEncoder` and `MultiBasicEncoder` used to process stereo pairs for feature matching and context generation. Structurally similar to RAFT extractors but integrated into the selective attention framework. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/selective_raft/raft.py`
[ Main Selective-RAFT architecture ]
[ Defines the `RAFT` class upgraded with `SelectiveConvGRU`, `SpatialAttentionExtractor`, and `ChannelAttentionEnhancement`. Enhances the baseline RAFT-Stereo iterative refinement process with adaptive, attention-gated recurrent updates. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/reference_impls/selective_raft/update.py`
[ Selective update machinery for Selective-RAFT ]
[ Contains the `SelectiveConvGRU` which dynamically selects between small and large kernel convolutions based on attention maps. Also includes `ChannelAttentionEnhancement` and `SpatialAttentionExtractor` to compute the gating signals, leading to faster convergence. ]

`file:///home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/techniques.md`
[ Catalog of reusable stereo vision techniques ]
[ An authoritative cheat sheet detailing common techniques across the stereo corpus. Covers Cost-Volume variants (Group-wise, Concatenation), Aggregation methods (3D hourglass, SGA), Output regression, Iterative refinement, Upsampling, and Edge-device specific architectural patterns. ]


`/home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/verified_numbers.md`
[ Pointer and single source of truth for benchmark numbers for literature methods ]
[ This markdown file is a reference pointer explaining that `papers/verified_performance.md` is the single source of truth for parameter counts, Scene Flow EPE, KITTI 2015 D1-all, and latency for various state-of-the-art stereo matching methods. It includes a quick reference table of verified numbers for 9 key methods (e.g., PSMNet, HITNet, RAFT-Stereo, FoundationStereo, etc.) and explicitly documents common "landmines" to avoid, such as ambiguous variant naming (HITNet L vs XL) or incorrect baseline numbers circulating in the literature. It establishes rules for adding new methods to the verified list. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/templates/evidence_assessment_template.md`
[ Markdown template for per-source evidence quality assessment ]
[ This template is used by the source verification agent to systematically evaluate the quality of research papers or sources. It provides a structured "Evidence Assessment Card" with sections for source identification, evidence level, publication venue quality (including predatory journal checks), author credibility, methodological quality, currency, and conflict of interest. It ends with an overall grade and recommendation on whether and how to use the source. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/templates/literature_matrix_template.md`
[ Markdown template for source-by-theme literature synthesis ]
[ This document provides a template for cross-tabulating sources against identified themes to assist in systematic evidence mapping. It contains structures for a Basic Matrix (tracking support/contradiction across themes), an Extended Matrix (including details like sample and method), an Evidence Convergence Summary to gauge overall strength and confidence, and a Gap Identification table. It is intended to be a living document for organizing evidence before writing a synthesis narrative. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/templates/preregistration_template.md`
[ Fill-in template for OSF Standard Pre-Data Collection Registration ]
[ This file is a standard 21-item preregistration template based on the OSF standard. It includes structured sections for Study Information, Design Plan, Sampling Plan, Variables (Manipulated, Measured, Indices), and Analysis Plan (Statistical Models, Inference Criteria, Data Exclusion, Exploratory Analyses). It acts as a guide to pre-specify study design and hypotheses prior to data collection. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/templates/prisma_protocol_template.md`
[ Template for systematic review protocols based on PRISMA-P 2015 ]
[ This comprehensive template helps researchers write systematic review protocols in compliance with PRISMA-P 2015 guidelines. It dictates administrative information, introduction/rationale, and detailed methods including eligibility criteria (PICOS), information sources, search strategy blocks, data management/selection processes, risk of bias assessment tools (RoB 2, ROBINS-I), data synthesis methods, and meta-bias assessment. It aims to be registered on PROSPERO or OSF prior to beginning a literature search. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/templates/prisma_report_template.md`
[ Template for systematic review reports following PRISMA 2020 ]
[ This template structures the final reporting of a systematic review or meta-analysis according to the PRISMA 2020 statement. It maps the 27 PRISMA items to corresponding sections, including a detailed abstract, introduction, methods (eligibility, search, data extraction, synthesis), results (including the standard PRISMA Flow Diagram, risk of bias tables, and certainty assessment like GRADE), and discussion. It serves as a strict guideline for finalizing review manuscripts. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/templates/research_brief_template.md`
[ Markdown template for concise, actionable research briefs ]
[ This file is a standard output format for "quick" mode research tasks. It structures the output into an Executive Summary, Background & Research Question, Key Findings (each with evidence strength and sources), Analysis & Implications (with actionable recommendations), Limitations, and References. It is designed to provide quick, evidence-based insights in 500-1500 words. ]

`/home/abrar/Research/stero_research_claude/CLAUDE.md`
[ Core guidance and status documentation for the Claude AI assistant ]
[ This document acts as the master guide and context file for Claude when working on this repository. It provides a high-level project overview (Paper Collection, Review Paper, StereoLite edge model), authors' affiliations, detailed directory architecture, and critical technical context including the evolution of stereo matching pipelines. Crucially, it documents the exact current status of the StereoLite architecture (0.874 M parameters, GhostConv, tile-hypothesis), the review paper's compile status, development commands, and an extensive list of "Things That Have Burned Me Before" (pitfalls in LaTeX, PDF parsing, model assumptions, and Modal/T4 execution) to prevent the AI from repeating past mistakes. ]

`/home/abrar/Research/stero_research_claude/README.md`
[ Top-level project README overviewing goals, paper analysis, and model design ]
[ This README provides a public-facing overview of the stereo vision research project. It outlines the three main phases: analyzing ~190 papers from an Awesome list, writing a comprehensive IEEE review paper, and building an edge-optimized stereo model (StereoLite). It summarizes paper analysis by priority, highlights key papers across categories (Foundation-Model, Iterative, Edge), outlines the directory structure, provides a field evolution timeline from 2002 to 2026, and details the edge model's design direction (combining efficient CNN encoders, fused cost volumes, and lite GRU updates) to achieve <50ms latency on edge hardware. ]

`/home/abrar/Research/stero_research_claude/awesome_list_raw.md`
[ Raw markdown source of the Awesome-Deep-Stereo-Matching repository ]
[ This is a raw copy of the "Awesome-Deep-Stereo-Matching" curated list maintained by Fabio Tosi and others. It catalogs hundreds of papers on deep stereo matching organized by categories such as Real-World/Synthetic Datasets, Learning for Stereo Pipelines (Matching Cost, Optimization, Refinement), End-to-End Architectures, Challenges, and Applications. It serves as the primary source material for the project's paper collection and review paper efforts. ]

`/home/abrar/Research/stero_research_claude/data/stereolite_v8_kaggle/load_pth_example.py`
[ Minimal inference script to test the exported Kaggle StereoLite v8 model ]
[ This Python script provides a minimal working example to load the `stereolite_v8_best.pth` checkpoint and run inference on a pair of stereo images. It initializes the StereoLite model, loads the weights, processes left and right input images (resizing and normalizing them), computes the disparity map without gradients, and outputs a color-mapped (TURBO) visualization of the disparity to a PNG file. ]

`/home/abrar/Research/stero_research_claude/data/stereolite_v8_kaggle/src/_blocks.py`
[ Shared neural network building blocks for the StereoLite architecture ]
[ This module contains the foundational neural network blocks used in the Kaggle build of StereoLite. Key components include `GhostConv` (which halves parameters using cheap depthwise convolutions), `SqueezeExcitation` (channel attention), `RepVGGBlock` (train-time multi-branch, deploy-time 3x3 fusion), `NeighborhoodAttention2d` (linear tile-propagation attention), and `SelectiveScan1d` (a pure-PyTorch implementation of a simplified Mamba-S6 scan over spatial dimensions). All blocks use `GroupNorm` instead of BatchNorm to ensure consistent train/eval behavior independent of batch size. ]

`/home/abrar/Research/stero_research_claude/data/stereolite_v8_kaggle/src/d1_tile/__init__.py`
[ Package initializer for the Kaggle StereoLite d1_tile module ]
[ A simple `__init__.py` file that exposes `StereoLite` and `StereoLiteConfig` from `model.py` for the Kaggle build of StereoLite v8. ]

`/home/abrar/Research/stero_research_claude/data/stereolite_v8_kaggle/src/d1_tile/model.py`
[ Main StereoLite v7/v8 network architecture implementation ]
[ This file defines the core `StereoLite` model, a highly efficient tile-hypothesis propagation stereo network aiming for edge deployment (~1.2-1.5M params). It includes the `TileFeatureEncoder` (GhostConv-based) and `MobileNetV2Encoder` backbones, `ConvexUpsample` for final refinement, and the `StereoLite` assembly which builds features at 1/2 to 1/16 scales, initializes a coarse cost volume at 1/16, and iteratively refines and upsamples disparity tiles through `TileRefine` and `TileUpsample` heads without heavy 3D convolutions at finer scales. It also defines the `StereoLiteConfig` dataclass to control iterations, channels, and backbones. ]

`/home/abrar/Research/stero_research_claude/data/stereolite_v8_kaggle/src/d1_tile/tile_propagate.py`
[ HITNet-inspired tile-hypothesis propagation modules for StereoLite ]
[ This module implements the core mechanics of tile-based disparity propagation. It defines the `TileState` dataclass storing disparity, x/y slopes, features, and confidence. It includes `TileInit` to bootstrap the initial state via a tiny 3D cost volume at 1/16 scale, `TileRefine` to update the state using warped right-image features and predicting residual updates, and `TileUpsample` to propagate the plane hypotheses to finer resolutions using the plane equation (incorporating slopes for sub-pixel accuracy). ]

`/home/abrar/Research/stero_research_claude/data/stereolite_v8_kaggle/src/sceneflow_loader.py`
[ Dataset loader for the Scene Flow Driving subset ]
[ This script handles loading and parsing the synthetic Scene Flow Driving dataset. It includes a custom PFM file reader to load ground truth disparity maps, a function to enumerate left/right/disparity triplets across forward/backward and slow/fast directories, a train/val split utility, and a PyTorch `Dataset` class (`SceneFlowDriving`) that loads and resizes the RGB images and proportionally resizes the ground truth disparities, returning them as tensors suitable for training. ]

`/home/abrar/Research/stero_research_claude/model/ARCHITECTURE.md`
[ Reference documentation for a legacy 11M parameter RAFT-Stereo checkpoint ]
[ This file describes `latest.pth`, an 11.12 M parameter checkpoint based on RAFT-Stereo. It explicitly documents that this massive architecture is an abandoned chassis and is NOT the foundation for new edge stereo work because it exceeds the target edge budget (0.6–3 M params) by an order of magnitude. The document preserves training metadata, hyperparameters, architecture shape, and instructions on how to use it merely as an optional distillation teacher or baseline sanity check, rather than a starting point for new models. ]

`/home/abrar/Research/stero_research_claude/model/README.md`
[ Overview and status of the edge-stereo model development ]
[ This README explains the pivot away from the heavy 11M parameter RAFT-Stereo baseline toward much lighter paradigms suitable for edge devices. It documents the directory structure and outlines three candidate designs: d1_tile (HITNet-based), d2_cascade (BGNet-based), and d3_sgm (GA-Net-based), emphasizing a strict budget of 0.6-3M parameters. It clarifies that all RAFT wrapper code was deleted to avoid anchoring on the wrong chassis and tracks the implementation status of these new candidates and their data loaders. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/EXPERIMENTS.md`
[ Chronological log of model overfitting, ablation, and training runs ]
[ This file is an automatically generated benchmark log tracking every experiment run, ordered newest to oldest. It logs the type of experiment (e.g., YOLO encoder ablation, loss ablation, architecture A/B overfits), configuration, variant names, parameter counts, and comprehensive metrics (EPE, RMSE, median error, bad-pixel percentages, D1-all, and latency). It serves as a historical record of what configurations worked, documenting the performance of various backbone choices and loss function combinations on small overfitting sets. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/OVERFIT_METHODOLOGY.md`
[ Methodology document for the 10-pair overfitting smoke test ]
[ This document outlines the rationale, setup, and interpretation of the overfitting harness used to quickly sanity-check new architectures before committing to full training. It details the 20-pair (or 10-pair) setup on Scene Flow Driving, the suite of benchmark metrics (EPE, RMSE, bad-1, D1-all), the composition of the multi-term loss function (multi-scale L1 + gradient consistency + bad-1 hinge), and exactly what constitutes "success" (EPE < 0.5px, no OOM) or "failure" (NaNs, flat loss) to rapidly identify bugs in gradient flow or model capacity. ]


`/home/abrar/Research/stero_research_claude/model/benchmarks/__init__.py`
[ Empty Python package initializer ]
[ An empty file used to mark the `benchmarks` directory as a Python package, allowing its modules to be imported. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/ablation_phase1_20260502-142629/ablation_phase1/baseline_n100_ghostconv/README.md`
[ Ablation log for Phase 1 GhostConv baseline on 100 pairs ]
[ Records the results of overfitting the Phase 1 `costlookup` architecture with a GhostConv encoder on 100 Scene Flow pairs. It achieved an EPE of 0.7658px and 15.10% bad-1.0 error with 1.25M parameters, running at ~25.36ms inference latency on an A100. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/ablation_phase2_20260502-152052/ablation_phase2/p2_baseline/README.md`
[ Ablation log for Phase 2 control baseline ]
[ Records the control baseline for the Phase 2 ablation sweep. It achieved an EPE of 0.8644px and 16.69% bad-1.0 error with 1.25M parameters and ~25.8ms inference latency on an A100 across 9000 steps. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/ablation_phase2_20260502-152052/ablation_phase2/p2_cascade_cv_4/README.md`
[ Ablation log for Phase 2 cascade_cv_4 variant ]
[ Records the performance of adding a narrow-range 3D cost volume between TileRefine iterations. It achieved an EPE of 0.8385px and 16.07% bad-1.0 error (a winning configuration in Phase 2) with 1.32M parameters, but increased latency to ~41.16ms. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/ablation_phase2_20260502-152052/ablation_phase2/p2_conf_aware/README.md`
[ Ablation log for Phase 2 conf_aware variant ]
[ Records the performance of a confidence-aware formulation. It achieved an EPE of 0.8718px and 16.36% bad-1.0 error with 1.25M parameters and ~25.19ms latency. It failed to beat the baseline. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/ablation_phase2_20260502-152052/ablation_phase2/p2_context_branch/README.md`
[ Ablation log for Phase 2 context_branch variant ]
[ Records the performance of adding a dedicated context encoder branch. It achieved an EPE of 0.8818px (worse than baseline) and 16.88% bad-1.0 error with 1.49M parameters and ~41.5ms latency. This intervention underperformed. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/ablation_phase2_20260502-152052/ablation_phase2/p2_edge_smooth/README.md`
[ Ablation log for Phase 2 edge_smooth variant ]
[ Records the performance of adding edge-aware smoothness loss. It achieved an EPE of 0.8734px and 16.44% bad-1.0 error with 1.25M parameters and ~25.6ms latency. It tied or slightly lost to the baseline. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/ablation_phase2_20260502-152052/ablation_phase2/p2_selective_gate/README.md`
[ Ablation log for Phase 2 selective_gate variant ]
[ Records the performance of a selective gating mechanism. It achieved an EPE of 0.8623px and 16.36% bad-1.0 error with 1.25M parameters and ~38.8ms latency, providing negligible accuracy gains at a steep latency cost. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/ablation_phase2_20260502-152052/ablation_phase2/p2_seq_loss/README.md`
[ Ablation log for Phase 2 seq_loss variant ]
[ Records the performance of a RAFT-style sequence loss. It achieved an EPE of 0.8936px and 16.79% bad-1.0 error with 1.25M parameters and ~47.4ms latency, surprisingly underperforming the baseline. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/ablation_phase2_20260502-152052/ablation_phase2/p2_slope_aware_warp/README.md`
[ Ablation log for Phase 2 slope_aware_warp variant ]
[ Records the performance of a slope-corrected sampling mechanism. It achieved an EPE of 0.8435px and 16.03% bad-1.0 error with 1.25M parameters and ~39.7ms latency, marking it as the second successful candidate in Phase 2. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/ablation_phase2_20260502-152052/ablation_phase2/p2_slope_sup/README.md`
[ Ablation log for Phase 2 slope_sup variant ]
[ Records the performance of explicit slope supervision. It achieved an EPE of 0.8972px and 17.25% bad-1.0 error with 1.25M parameters and ~26.8ms latency, significantly underperforming the baseline. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/ablation_phase2_REPORT.md`
[ Comprehensive analysis report for the 9-variant Phase 2 sweep ]
[ This markdown report synthesizes the results of 9 parallel architecture ablations run on A100 instances. It includes a full result table, delta comparisons against the baseline, and detailed findings. It identifies `cascade_cv_4` and `slope_aware_warp` as the only two winning variants, dismisses loss-side modifications like `seq_loss` and `slope_sup` as detrimental, and charts a course for a Phase 3 composition of the two winning features. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/arch_ablation_20260501-122438/current/README.md`
[ Ablation log for the 'current' architecture baseline on 20 pairs ]
[ Records an early 3000-step overfitting run on an RTX 3050 Laptop GPU for the 'current' (0.53M parameter) model variant. It achieved 0.5776px EPE and 10.72% bad-1.0 error with ~23.7ms inference latency. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/arch_ablation_20260501-122438/v1_iter/README.md`
[ Ablation log for the 'v1_iter' architecture variant on 20 pairs ]
[ Records an early 3000-step overfitting run on an RTX 3050 Laptop GPU for the 'v1_iter' (0.57M parameter) model variant. It achieved 0.5729px EPE and 11.12% bad-1.0 error with ~32.4ms inference latency. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/arch_ablation_20260501-122438/v2_hitnet/README.md`
[ Ablation log for the 'v2_hitnet' architecture variant on 20 pairs ]
[ Records an early 3000-step overfitting run on an RTX 3050 Laptop GPU for the 'v2_hitnet' (0.48M parameter) model variant. It achieved 0.8444px EPE and 19.30% bad-1.0 error with ~25.2ms inference latency. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_baseline/README.md`
[ Gamma sweep re-run log for Phase 2 control baseline ]
[ Records the baseline results during a gamma parameter sweep, replicating the 0.8644px EPE metrics of the Phase 2 baseline run. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_cascade_cv_4/README.md`
[ Gamma sweep re-run log for Phase 2 cascade_cv_4 variant ]
[ Records the results for the cascade_cv_4 variant during a gamma sweep, maintaining the 0.8385px EPE performance found in Phase 2. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_conf_aware/README.md`
[ Gamma sweep re-run log for Phase 2 conf_aware variant ]
[ Records the results for the conf_aware variant during a gamma sweep, showing 0.8718px EPE. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_context_branch/README.md`
[ Gamma sweep re-run log for Phase 2 context_branch variant ]
[ Records the results for the context_branch variant during a gamma sweep, showing 0.8818px EPE. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_edge_smooth/README.md`
[ Gamma sweep re-run log for Phase 2 edge_smooth variant ]
[ Records the results for the edge_smooth variant during a gamma sweep, showing 0.8734px EPE. ]


`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_selective_gate/README.md`
[ Gamma sweep re-run log for Phase 2 selective_gate variant ]
[ Records the results for the selective_gate variant during a gamma sweep, replicating the 0.8623px EPE from Phase 2. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_seq_loss/README.md`
[ Gamma sweep re-run log for Phase 2 seq_loss variant ]
[ Records the results for the seq_loss variant during a gamma sweep, showing 0.8936px EPE. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_seq_loss_g05/README.md`
[ Gamma sweep ablation: seq_loss with gamma=0.5 ]
[ Records the result of testing the seq_loss formulation with a gamma weight of 0.5. It achieved 0.8417px EPE and 15.79% bad-1.0 error, showing that a lower gamma weight significantly improves seq_loss performance compared to gamma=0.9. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_seq_loss_g06/README.md`
[ Gamma sweep ablation: seq_loss with gamma=0.6 ]
[ Records the result of testing the seq_loss formulation with a gamma weight of 0.6. It achieved 0.8514px EPE and 16.01% bad-1.0 error. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_seq_loss_g07/README.md`
[ Gamma sweep ablation: seq_loss with gamma=0.7 ]
[ Records the result of testing the seq_loss formulation with a gamma weight of 0.7. It achieved 0.8752px EPE and 16.39% bad-1.0 error. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_slope_aware_warp/README.md`
[ Gamma sweep re-run log for Phase 2 slope_aware_warp variant ]
[ Records the results for the slope_aware_warp variant during a gamma sweep, showing 0.8435px EPE. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/gamma_sweep_20260502-162822/ablation_phase2/p2_slope_sup/README.md`
[ Gamma sweep re-run log for Phase 2 slope_sup variant ]
[ Records the results for the slope_sup variant during a gamma sweep, showing 0.8972px EPE. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/latency.py`
[ Python script for measuring model inference latency ]
[ This script provides a harness to profile wall-clock inference latency for stereo architectures. It generates random tensors for left and right images, runs a short warm-up, and then times multiple inference trials, reporting median and p95 latency along with a frame rate (FPS) estimate. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/loss_ablation_20260501-132948/L1/README.md`
[ Ablation log for simple L1 loss variant ]
[ Records the results of using a pure multi-scale L1 loss on the early 'current' (GhostConv) architecture over 20 pairs. It achieved an EPE of 0.6629px and 12.10% bad-1.0 error. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/loss_ablation_20260501-132948/L1_bad1/README.md`
[ Ablation log for L1 loss augmented with a bad-1.0 hinge term ]
[ Records the results of using L1 loss combined with a differentiable approximation of the bad-1.0 metric. It achieved a worse EPE of 0.7631px and 18.59% bad-1.0 error. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/loss_ablation_20260501-132948/L1_grad/README.md`
[ Ablation log for L1 loss augmented with gradient consistency ]
[ Records the results of combining multi-scale L1 loss with a spatial gradient consistency term. It achieved a strong EPE of 0.6178px and 10.73% bad-1.0 error, proving the value of gradient supervision. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/loss_ablation_20260501-132948/L1_seq/README.md`
[ Ablation log for sequence L1 loss ]
[ Records the results of applying sequence L1 loss. It performed poorly, with 1.0053px EPE and 28.65% bad-1.0 error, significantly underperforming simple L1. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/loss_ablation_20260501-132948/charbonnier/README.md`
[ Ablation log for Charbonnier loss variant ]
[ Records the results of using a pseudo-Huber (Charbonnier) loss. It achieved 0.6309px EPE and 11.96% bad-1.0 error, showing decent performance. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/loss_ablation_20260501-132948/cocktail/README.md`
[ Ablation log for cocktail loss (L1 + gradient + SSIM) ]
[ Records the results of combining L1, gradient consistency, and structural similarity (SSIM). It achieved 0.6915px EPE, slightly underperforming L1+grad alone. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/loss_ablation_20260501-132948/cocktail_b05/README.md`
[ Ablation log for cocktail loss with bad-0.5 hinge ]
[ Records the results of adding a bad-0.5 hinge to the cocktail loss. It achieved 0.6765px EPE. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/loss_ablation_20260501-132948/stack/README.md`
[ Ablation log for stack loss (L1 + grad + bad-1 hinge) ]
[ Records the results of combining L1, gradient consistency, and a bad-1.0 hinge penalty. It achieved 0.6716px EPE. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/loss_ablation_20260501-132948/stack_d1/README.md`
[ Ablation log for stack loss augmented with D1 hinge ]
[ Records the results of adding a D1-error approximation hinge to the stack loss. It achieved a very strong 0.5913px EPE and 11.73% bad-1.0 error, making it a highly effective loss formulation. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-214720/README.md`
[ Comparative report on the matched overfit experiments for 3 encoders ]
[ This markdown report compares three candidate backbone encoders (GhostConv, YOLO26n, and YOLO26s) overfitting on 10 fixed Scene Flow pairs. It charts param count vs. latency vs. final EPE. It concludes that wider YOLO channels converge faster and preserve fine canopy details much better than GhostConv, despite GhostConv having faster latency. It ultimately recommends YOLO26n as the best balance of speed and memorization capacity for the next phase. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-214720/ghost/README.md`
[ Log of the GhostConv encoder overfitting run ]
[ Records the detailed results for the GhostConv backbone variant during the matched overfit comparison (0.538M params, 0.5929px EPE, 23.1ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-214720/yolo26n/README.md`
[ Log of the YOLO26n encoder overfitting run ]
[ Records the detailed results for the YOLO26n backbone variant during the matched overfit comparison (0.808M params, 0.5508px EPE, 25.25ms latency). ]


`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-214720/yolo26s/README.md`
[ Log of the YOLO26s encoder overfitting run on 10 pairs ]
[ Records the results for the YOLO26s backbone variant during the first matched overfit comparison. It achieved the best performance with 0.4392px EPE and 31.26ms latency, albeit with 2.06M parameters. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-225114/README.md`
[ Comparative report on the matched overfit experiments for 3 encoders on the full metric suite ]
[ This markdown report compares GhostConv, YOLO26n, and YOLO26s backbones on 10 fixed Scene Flow pairs, but this time reporting the full benchmark suite (EPE, RMSE, median AE, bad-0.5/1/2/3, D1-all). It highlights that EPE alone was misleading, and that YOLO26s consistently outperforms the other variants on all metrics, proving it captures fine detail much better than GhostConv. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-225114/ghost/README.md`
[ Log of the GhostConv encoder overfitting run on 10 pairs (full metrics) ]
[ Records the detailed results for the GhostConv backbone variant during the full-metric overfit comparison (0.538M params, 0.696px EPE, 11.49% bad-1.0, 23.99ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-225114/yolo26n/README.md`
[ Log of the YOLO26n encoder overfitting run on 10 pairs (full metrics) ]
[ Records the detailed results for the YOLO26n backbone variant during the full-metric overfit comparison (0.808M params, 0.7836px EPE, 29.83% bad-1.0, 24.93ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-225114/yolo26s/README.md`
[ Log of the YOLO26s encoder overfitting run on 10 pairs (full metrics) ]
[ Records the detailed results for the YOLO26s backbone variant during the full-metric overfit comparison (2.06M params, 0.3751px EPE, 6.19% bad-1.0, 29.65ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-234721/README.md`
[ Comparative report on the reproducible matched overfit on 20 pairs ]
[ This markdown report compares the three candidate encoders (GhostConv, YOLO26n, and YOLO26s) on an upgraded methodology: perfectly reproducible runs, 20 fixed pairs, and a multi-term loss (L1 + grad + bad-1 hinge). It confirms that YOLO26s is the unambiguous winner on accuracy, while GhostConv remains the winner for speed and memory efficiency. YOLO26n is seen as dominated. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-234721/ghost/README.md`
[ Log of the GhostConv encoder reproducible overfitting run ]
[ Records the detailed results for the GhostConv backbone variant during the reproducible 20-pair overfit comparison (0.6245px EPE, 13.31% bad-1.0, 23.52ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-234721/yolo26n/README.md`
[ Log of the YOLO26n encoder reproducible overfitting run ]
[ Records the detailed results for the YOLO26n backbone variant during the reproducible 20-pair overfit comparison (0.7121px EPE, 18.07% bad-1.0, 24.12ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/matched_overfit_20260430-234721/yolo26s/README.md`
[ Log of the YOLO26s encoder reproducible overfitting run ]
[ Records the detailed results for the YOLO26s backbone variant during the reproducible 20-pair overfit comparison (0.5283px EPE, 10.88% bad-1.0, 25.83ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/phase3_combined_20260502-165700/ablation_phase3/p3_cascade_plus_slopewarp/README.md`
[ Ablation log for Phase 3 combined variant ]
[ Records the results of combining the two winning mechanisms from Phase 2 (`cascade_cv_4` and `slope_aware_warp`). It achieved 0.8246px EPE and 15.85% bad-1.0 error with a 53.1ms latency on a T4. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/REPORT.md`
[ Comprehensive report for the 12-config architecture sweep ]
[ This markdown report analyzes a sweep of 12 configurations spanning 3 mechanics (costlookup, tilegru, raftlike), 2 encoders (GhostConv, YOLO26n), and 2 phases (P1 with ConvexUpsample vs P2 with extended TileRefine). It discovers that the Phase 2 chassis unlocks YOLO26n's performance, making `costlookup_yolo26n_full` the best overall config (0.598px EPE), and establishes that more architecture isn't always better (combinations often backfire). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/costlookup_ghost/README.md`
[ Ablation log for Phase 1 costlookup_ghost variant ]
[ Records the results of the Phase 1 costlookup_ghost model (0.59M params, 0.5743px EPE, 39.9ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/costlookup_ghost_full/README.md`
[ Ablation log for Phase 2 costlookup_ghost_full variant ]
[ Records the results of the Phase 2 costlookup_ghost_full model, which extended TileRefine to 1/2 resolution (0.64M params, 0.6102px EPE, 50.8ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/costlookup_yolo26n/README.md`
[ Ablation log for Phase 1 costlookup_yolo26n variant ]
[ Records the results of the Phase 1 costlookup_yolo26n model (0.86M params, 0.7500px EPE, 36.4ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/costlookup_yolo26n_full/README.md`
[ Ablation log for Phase 2 costlookup_yolo26n_full variant ]
[ Records the standout results of the Phase 2 costlookup_yolo26n_full model, which achieved 0.5984px EPE and 10.33% bad-1.0 error with 49.8ms latency, making it the top candidate. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/raftlike_ghost/README.md`
[ Ablation log for Phase 1 raftlike_ghost variant ]
[ Records the results of the Phase 1 raftlike_ghost model (0.54M params, 0.7966px EPE, 41.8ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/raftlike_ghost_full/README.md`
[ Ablation log for Phase 2 raftlike_ghost_full variant ]
[ Records the results of the Phase 2 raftlike_ghost_full model (0.58M params, 0.6141px EPE, 56.8ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/raftlike_yolo26n/README.md`
[ Ablation log for Phase 1 raftlike_yolo26n variant ]
[ Records the results of the Phase 1 raftlike_yolo26n model (0.81M params, 0.6246px EPE, 41.8ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/raftlike_yolo26n_full/README.md`
[ Ablation log for Phase 2 raftlike_yolo26n_full variant ]
[ Records the results of the Phase 2 raftlike_yolo26n_full model (0.84M params, 0.7303px EPE, 59.7ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/tilegru_ghost/README.md`
[ Ablation log for Phase 1 tilegru_ghost variant ]
[ Records the results of the Phase 1 tilegru_ghost model (0.49M params, 0.6779px EPE, 27.6ms latency). ]


`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/tilegru_ghost_full/README.md`
[ Ablation log for Phase 2 tilegru_ghost_full variant ]
[ Records the results of the Phase 2 tilegru_ghost_full model (0.52M params, 0.6402px EPE, 40.3ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/tilegru_yolo26n/README.md`
[ Ablation log for Phase 1 tilegru_yolo26n variant ]
[ Records the results of the Phase 1 tilegru_yolo26n model (0.76M params, 0.6132px EPE, 29.3ms latency). Fast and balanced. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/raftlike_sweep_20260501-211601/tilegru_yolo26n_full/README.md`
[ Ablation log for Phase 2 tilegru_yolo26n_full variant ]
[ Records the results of the Phase 2 tilegru_yolo26n_full model (0.78M params, 0.7764px EPE, 37.5ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/widener_n100_20260502-013518/yolo26s_native_ceiling/README.md`
[ Log of the YOLO26s native ceiling run on 100 pairs ]
[ Records the baseline performance of the YOLO26s backbone (2.13M params, 1.039px EPE, 19.22% bad-1.0, 56.1ms latency) on 100 pairs, serving as the ceiling target for the subsequent feature-widener ablation sweep. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/yolo_ablation_20260430-202931/yolo26n/README.md`
[ Initial YOLO26n ablation log ]
[ Initial smoke test of YOLO26n in StereoLite (0.81M params, 0.5763px EPE). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/yolo_ablation_20260430-204526/README.md`
[ Report on YOLO26n vs YOLO26s overfit ablation ]
[ Summarizes the side-by-side smoke test of YOLO26n and YOLO26s plugged into the StereoLite pipeline. Validates that both can overfit 10 pairs successfully (0.625px and 0.459px EPE respectively) and run with acceptable latency (~24.6ms and ~27.4ms) on a 3050. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/yolo_ablation_20260430-204526/yolo26n/README.md`
[ Log of the YOLO26n encoder overfitting run (smoke test) ]
[ Detailed results of the YOLO26n smoke test run (0.81M params, 0.6252px EPE, 24.57ms latency). ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/yolo_ablation_20260430-204526/yolo26s/README.md`
[ Log of the YOLO26s encoder overfitting run (smoke test) ]
[ Detailed results of the YOLO26s smoke test run (2.06M params, 0.4592px EPE, 27.4ms latency). ]

`/home/abrar/Research/stero_research_claude/model/data/__init__.py`
[ Dataset loader stubs ]
[ Python init file containing docstrings outlining the planned datasets: InStereo2K, KITTI 2015, Scene Flow, Middlebury v3, KITTI Raw, and DrivingStereo. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/README.md`
[ Documentation for the StereoLite v9 architecture ]
[ Main README for the StereoLite edge-tier model using GhostConv/MobileNetV2 encoders. Describes the pipeline (tile-hypothesis with iterative refinement, plane-equation upsampling, learned convex upsample) and lists current stats (0.87M params, ~23.5ms latency, 1.54px best val EPE on InStereo2K). ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/__init__.py`
[ StereoLite package init ]
[ Imports StereoLite and StereoLiteConfig from model.py. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/arch_refs/README.md`
[ Curated list of reference stereo architecture diagrams ]
[ A markdown file categorizing 17 "Tier A" stereo network architecture diagrams from literature (e.g., RAFTStereo, DEFOM, IGEV, HITNet) to serve as inspiration for rendering the final StereoLite v9 publication diagram. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/cost_volume.py`
[ Implementation of cost volume logic for StereoLite ]
[ Defines `GroupwiseCostVolume1D8`, a group-wise correlation module with a 1-level 3D hourglass aggregator running at 1/8 scale, featuring CoEx-style Guided Cost Excitation. Also includes `CascadeRefinementVolume` for narrow-range cascade refinements. Heavily utilizes INT8-friendly GroupNorm + SiLU 3D convolutions. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/draw_arch_comparison.py`
[ Script to generate side-by-side architecture diagram comparing HITNet to StereoLite v9 ]
[ Uses matplotlib to draw a detailed flowchart comparing the HITNet architecture (multi-scale tile init) with StereoLite v9 (MobileNetV2, single-scale init, recurrent iterative updates, convex upsampling cascade). Generates PDF/PNG. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/draw_deployment_pipeline.py`
[ Script to generate edge deployment pipeline diagram ]
[ Uses matplotlib to draw a vertical block diagram showing the data flow from AR0144 stereo USB camera -> Jetson Nano -> StereoLite (INT8/FP16) -> Socket Stream output. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/draw_mobilenet_truncation.py`
[ Script to generate MobileNetV2 truncation explanation diagram ]
[ Uses matplotlib to draw a 3-panel diagram explaining the memory/param optimization applied to timm's MobileNetV2, where unused deep blocks (blocks 5 & 6) are explicitly deleted to save compute and parameters (2.14M -> 0.874M). ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/draw_stereolite_arch.py`
[ Script to generate publication-quality StereoLite architecture diagram ]
[ Uses matplotlib to draw a highly detailed, colored, and annotated block diagram of the StereoLite v9 architecture, incorporating image thumbnails, cost volume prisms, iterative refinement loops, and a legend. Designed based on Tier A reference papers. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/make_training_figures.py`
[ Script to generate training result figures ]
[ Parses Kaggle training logs (`train_log.csv`) and sample images to generate a 4-panel training curve figure (loss, L1, LR), a progress grid tracking two pairs across epochs, and a final gallery of 6 val pairs. Outputs PDFs and PNGs. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/model.py`
[ Core StereoLite v7/v9 PyTorch model implementation ]
[ Defines the `StereoLite` nn.Module and `StereoLiteConfig`. Includes the `TileFeatureEncoder` (custom GhostConv) and `MobileNetV2Encoder` (truncated timm backbone), along with the `ConvexUpsample` logic. The `forward` pass constructs the pipeline: init -> iterate -> plane upsample -> iterate -> plane upsample -> iterate -> convex upsample. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/tile_propagate.py`
[ Implementation of tile-hypothesis propagation modules ]
[ Defines the `TileState` dataclass. Implements `TileInit` (tiny local CV + soft-argmin), `TileRefine` (recurrent warp-regress update using plane equation), and `TileUpsample` (applies plane equation d + sx*dx + sy*dy to upsample to a denser grid). ]


`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_costlookup/__init__.py`
[ Package init ]
[ Empty package init. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_costlookup/model.py`
[ Architecture variant: costlookup ]
[ Adds RAFT-style per-iteration local cost lookup to the TileRefine loop. Includes the `extend_to_full` phase 2 configuration and Phase-2 ablation knobs (`slope_aware_warp`, `selective_gate`, `cascade_cv_4`, `context_branch`). ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_costlookup/tile_propagate.py`
[ TilePropagate with cost lookup ]
[ Modifies `TileRefine` into `TileRefineCostLookup`. At each iteration, computes a `(2*half_range+1)`-slice groupwise correlation around the current disparity and concatenates it to the `TileRefine` input. Introduces ablation knobs like `selective_gate` and `slope_aware_warp`. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_raftlike/__init__.py`
[ Package init ]
[ Empty package init. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_raftlike/model.py`
[ Architecture variant: raftlike ]
[ Combines local cost lookup with ConvGRU on the tile feature slot. Represents the closest tile-resolution analogue of RAFT-Stereo. Allows extending to full resolution with Phase 2 mechanism. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_raftlike/tile_propagate.py`
[ TilePropagate with cost lookup and GRU ]
[ Implements `TileRefineRAFTLike`. Integrates `_correlation_lookup` around current disparity, concatenates the resulting cost with the context input, and updates the hidden state (tile feature) using ConvGRU gating operations. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_tilegru/__init__.py`
[ Package init ]
[ Empty package init. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_tilegru/model.py`
[ Architecture variant: tilegru ]
[ Replaces the stateless 3-layer `TileRefine` trunk with a ConvGRU. Propagates the GRU hidden state across iterations and scales. Can extend to full resolution (`extend_to_full`). ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_tilegru/tile_propagate.py`
[ TilePropagate with GRU ]
[ Implements `TileRefineGRU`. Modifies the refinement to use ConvGRU operations on the tile feature state. Propagates state context across scales via the plane-equation upsampling mechanism. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_v1_iter/__init__.py`
[ Package init ]
[ Imports StereoLite and StereoLiteConfig from model.py. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_v1_iter/cost_volume.py`
[ Cost volume for v1_iter ]
[ Defines `GroupwiseCostVolume1D8` and `CascadeRefinementVolume`. Identical to the edge tier cost volume components. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_v1_iter/model.py`
[ Architecture variant: v1_iter ]
[ Variant A experiment. Removes `ConvexUpsample` entirely. Relies solely on plane-equation upsampling across all scales including the final step. Adds iterations at 1/2 resolution to verify if continuous iteration outperforms learned mask upsampling. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_v1_iter/tile_propagate.py`
[ Tile propagation for v1_iter ]
[ Contains `TileInit`, `TileRefine`, and `TileUpsample` identical to the baseline pipeline. Used to test the fully iterative variant without `ConvexUpsample`. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_v2_hitnet/__init__.py`
[ Package init ]
[ Imports StereoLite and StereoLiteConfig from model.py. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_v2_hitnet/cost_volume.py`
[ Cost volume for v2_hitnet ]
[ Defines `GroupwiseCostVolume1D8` and `CascadeRefinementVolume`. Matches the standard implementation. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_v2_hitnet/hitnet_propagate.py`
[ HITNet exact propagation block ]
[ Implements the `HITNetPropagate` block and `HITNetResBlock`. Adapts the HITNet per-scale propagation mechanism (single-pass, dilated convs, no BN, augmented input with 3 disparity offsets) to the StereoLite feature map resolution. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_v2_hitnet/model.py`
[ Architecture variant: v2_hitnet ]
[ Variant B experiment. A faithful adaptation of the HITNet architecture. Uses single-pass `HITNetPropagate` blocks per scale and plane-equation upsampling without any `ConvexUpsample` modules. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_v2_hitnet/tile_propagate.py`
[ Tile propagation for v2_hitnet ]
[ Standard `TileInit` and `TileUpsample` operations for the v2_hitnet variant. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/README.md`
[ YOLO variant documentation ]
[ Documents the "mid tier" StereoLite variant substituting GhostConv for a truncated YOLO26s or YOLO26n backbone. Aimed at devices like Jetson Orin Nano with 4-6 TOPS, achieving 2.06M params and ~25.8ms latency. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/__init__.py`
[ Package init ]
[ Imports StereoLite and StereoLiteConfig from model.py. ]


`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/arch_refs/README.md`
[ Architecture diagram references ]
[ Documents design principles for the StereoLite v9 architecture diagram, drawing inspiration from 17 TIER-A stereo papers like RAFT-Stereo, IGEV, and HITNet. Gives examples of good visualizations like 3D isometric conv prisms, correlation pyramids, etc. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/cost_volume.py`
[ Cost volume for YOLO variant ]
[ Defines `GroupwiseCostVolume1D8` and `CascadeRefinementVolume`, very similar to edge tier cost volume. Implements group-wise correlation with 1-level 3D hourglass aggregator. Adds `GuidedCostExcitation` for channel-wise gate on 3D aggregator features. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/draw_arch_comparison.py`
[ Architecture comparison diagram script ]
[ Python script to render a side-by-side HITNet vs StereoLite-v8 (or v9) architecture diagram. Contains visual blocks and arrows describing the two architectures, and a table summarizing differentiators. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/draw_deployment_pipeline.py`
[ Deployment pipeline diagram script ]
[ Python script to render a vertical deployment-pipeline diagram showing AR0144 stereo USB camera -> Jetson Nano -> StereoLite -> Socket stream. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/draw_mobilenet_truncation.py`
[ MobileNetV2 truncation diagram script ]
[ Python script to render a 3-panel diagram explaining the MobileNetV2 truncation fix. Shows what was requested from timm, what timm actually built, and the fix of explicitly truncating unused blocks to save parameters and compute. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/draw_stereolite_arch.py`
[ StereoLite architecture diagram script ]
[ Python script to render a publication-quality architecture diagram for StereoLite. Draws input thumbnails, MobileNetV2 encoder, 3D cost volume, tile state, iterative refinement with ConvGRUs, and convex upsample blocks. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/make_training_figures.py`
[ Training figures generation script ]
[ Python script to generate figures from Kaggle training artifacts. Creates a 4-panel training curves plot, a progress grid showing pairs across checkpoints, and a final gallery of validation pairs. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/model.py`
[ Architecture variant: yolo ]
[ Defines `StereoLite` architecture, specifically v7. Supports "mobilenet", "yolo26n", "yolo26s", and "ghost" backbones. Implements iterative refinement over scales (1/16, 1/8, 1/4) with plane equation upsampling, and a final learned convex upsample. Uses `TileInit`, `TileRefine`, and `ConvexUpsample`. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/tile_propagate.py`
[ Tile propagation for YOLO variant ]
[ HITNet-inspired plane-tile hypothesis propagation. Defines `TileState` storing d, sx, sy, feat, conf. Implements `TileInit` (tiny 3D aggregator), `TileRefine` (warp-regress with Conv2D trunk), and `TileUpsample` (plane equation sub-pixel interpolation). ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/yolo_encoder.py`
[ YOLO truncated encoder ]
[ Implements `YoloTruncatedEncoder` which loads `yolo26n.pt` or `yolo26s.pt` via `ultralytics`. Slices the first 7 modules of the backbone to expose features at strides (2, 4, 8, 16) as required by StereoLite. Drop unused detection heads and FPN. ]

`/home/abrar/Research/stero_research_claude/model/designs/__init__.py`
[ Package init ]
[ Empty package init. ]

`/home/abrar/Research/stero_research_claude/model/designs/_blocks.py`
[ Shared building blocks ]
[ Contains `GhostConv`, `SqueezeExcitation`, `RepVGGBlock`, `NeighborhoodAttention2d`, and `SelectiveScan1d` (Mamba S6 core). Uses `GroupNorm` instead of `BatchNorm`. ]

`/home/abrar/Research/stero_research_claude/model/designs/_wideners.py`
[ Feature wideners ]
[ Implementations of various feature wideners to adapt YOLO26n features (which are thinner) to StereoLite. Includes `WidenerTier1` (1x1 adapters), `WidenerTier2DW`, `WidenerTier2MBConv`, `WidenerTier2Ghost`, `WidenerTier3TopDownFPN`, and `WidenerTier3BiFPN`. Also includes a function to replace BN with GN in place. ]

`/home/abrar/Research/stero_research_claude/model/evaluation/__init__.py`
[ Package init ]
[ Empty package init. ]

`/home/abrar/Research/stero_research_claude/model/evaluation/run_eval.py`
[ Evaluation script stub ]
[ Stub for an evaluation harness to compute EPE / bad-1 / bad-3 / D1-all across a dataset. Currently raises `NotImplementedError`. ]

`/home/abrar/Research/stero_research_claude/model/kaggle/build_notebook.py`
[ Kaggle notebook builder ]
[ Python script to build the Kaggle notebook `stereolite_v8_kaggle.ipynb`. Inlines source files (`_blocks.py`, `tile_propagate.py`, `model.py`, `sceneflow_loader.py`) and training scripts (`train_ddp.py`, `export.py`) into `%%writefile` cells for standalone execution on Kaggle. ]

`/home/abrar/Research/stero_research_claude/model/kaggle/stereolite_v8_kaggle.ipynb`
[ Kaggle Notebook ]
[ The generated notebook for Kaggle Training + Multi-format Export of StereoLite v8. ]

`/home/abrar/Research/stero_research_claude/model/scripts/README_pseudo_dataset_generation.md`
[ Pseudo-dataset generation docs ]
[ Documentation for running the FoundationStereo teacher model on a separate PC to produce pseudo-ground-truth disparity `.npy` files for training StereoLite. Explains setup, hardware requirements, running `run_teacher.py`, and troubleshooting. ]

`/home/abrar/Research/stero_research_claude/model/scripts/build_arch_mosaic.py`
[ Architecture mosaic builder ]
[ Python script to build comparison mosaics for the architecture sweep. Renders variant viz panels into grids for Phase 1, Phase 2, and a master sweep combining them. Extracts metrics from `meta.json`. ]

`/home/abrar/Research/stero_research_claude/model/scripts/build_experiments_summary.py`
[ Experiments summary builder ]
[ Python script to scan `model/benchmarks/` for `meta.json` files and generate a master `EXPERIMENTS.md` summary table containing final stereo metrics, parameters, and inference latency. ]


`/home/abrar/Research/stero_research_claude/model/scripts/build_loss_mosaic.py`
[ Loss comparison mosaic builder ]
[ Python script to build a comparison mosaic across different loss formulations. Probably renders outputs from different loss functions for visual comparison. ]

`/home/abrar/Research/stero_research_claude/model/scripts/build_loss_zoom.py`
[ Loss comparison zoom builder ]
[ Python script to build a zoomed-in comparison mosaic across different loss formulations. ]

`/home/abrar/Research/stero_research_claude/model/scripts/capture_interactive.py`
[ Interactive pair capture ]
[ Script for interactive stereo pair capture. ]

`/home/abrar/Research/stero_research_claude/model/scripts/capture_live_inference.py`
[ Live inference capture ]
[ Script for capturing live inference from a stereo camera. ]

`/home/abrar/Research/stero_research_claude/model/scripts/compare_student_vs_teacher.py`
[ Student vs teacher comparison ]
[ Python script to compare the StereoLite student model against the FoundationStereo teacher model, possibly outputting visual or numerical comparisons. ]

`/home/abrar/Research/stero_research_claude/model/scripts/disp_vis_regen.py`
[ Disparity visualization regenerator ]
[ Python script to regenerate disparity colormaps/visualizations from raw numpy files. ]

`/home/abrar/Research/stero_research_claude/model/scripts/disparity_to_pointcloud.py`
[ Disparity to pointcloud converter ]
[ Python script to convert disparity maps to 3D pointclouds using camera intrinsics. ]

`/home/abrar/Research/stero_research_claude/model/scripts/distill_train.py`
[ Distillation training script ]
[ Python script to train the StereoLite model via distillation using pseudo ground-truth generated by a teacher model. ]

`/home/abrar/Research/stero_research_claude/model/scripts/eval_sceneflow.py`
[ SceneFlow evaluation script ]
[ Python script to evaluate the StereoLite model on the SceneFlow dataset, computing EPE, D1-all, and other standard stereo metrics. ]

`/home/abrar/Research/stero_research_claude/model/scripts/hitnet_baseline.py`
[ HITNet baseline wrapper ]
[ Wrapper to load and run pretrained TinyHITNet (Scene Flow checkpoint) as a baseline for side-by-side comparisons. Handles HITNet's internal normalization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/infer_video_stereolite.py`
[ Video inference script ]
[ Run StereoLite on a stereo video pair and write a side-by-side [Left | colorized disparity] MP4 at the source frame rate. Computes fixed disparity colormap range with a warm-up pass. ]

`/home/abrar/Research/stero_research_claude/model/scripts/inspect_pseudo_dataset.py`
[ Pseudo dataset inspector ]
[ Quality scan over a FoundationStereo pseudo-GT dataset. Computes per-pair statistics, applies multi-criterion quality filters (corrupt, outlier, dark, uniform), and writes clean pairs and reports. ]

`/home/abrar/Research/stero_research_claude/model/scripts/live_stereolite.py`
[ Live stereo inference ]
[ Live stereo-camera inference for StereoLite v8. Opens CCB stereo camera, splits L/R, runs trained model, and shows a three-panel cv2 window. ]

`/home/abrar/Research/stero_research_claude/model/scripts/make_training_gif.py`
[ Training GIF maker ]
[ Build animated GIFs from a montage/training folder. Reads step frames, downscales, overlays a step counter, and writes training progression GIFs. ]

`/home/abrar/Research/stero_research_claude/model/scripts/overfit_arch_ablation.py`
[ Architecture ablation overfit harness ]
[ 3-way architecture A/B/C overfit comparison. Compares 'current' (yolo+ghost, ConvexUpsample), 'v1_iter' (more iterations, no ConvexUpsample), and 'v2_hitnet' (HITNet propagation block) on a small set. ]

`/home/abrar/Research/stero_research_claude/model/scripts/overfit_loss_ablation.py`
[ Loss ablation overfit harness ]
[ Loss-formulation A/B harness for the StereoLite chassis. Architecture is fixed. Evaluates L1, L1_seq, L1_grad, L1_bad1, cocktail, and cocktail_b05 formulations on a memorized set. ]

`/home/abrar/Research/stero_research_claude/model/scripts/overfit_yolo_ablation.py`
[ YOLO ablation overfit harness ]
[ Overfit StereoLite_yolo on 10 fixed Scene Flow Driving pairs. Sanity tests if YOLO26n/s truncated backbones wire up cleanly and can drive EPE down on a tiny memorized set. ]

`/home/abrar/Research/stero_research_claude/model/scripts/pseudo_pairs_loader.py`
[ Pseudo pairs dataset loader ]
[ Dataset loader for FoundationStereo pseudo-GT pairs. Drops into existing trainer. Reads the clean filter list. Handles disparity-on-resize math correctly. ]

`/home/abrar/Research/stero_research_claude/model/scripts/render_rotating_pc.py`
[ Rotating pointcloud renderer ]
[ Render rotating/panning/zooming GIFs of the model's predicted point cloud on validation pairs, using Open3D's OffscreenRenderer. ]

`/home/abrar/Research/stero_research_claude/model/scripts/run_teacher.py`
[ Teacher model inference script ]
[ Run FoundationStereo (ViT-Small variant) on saved stereo pairs and write per-pair pseudo-GT disparity to disk. Handles resizing logic so that pseudo-GT matches student resolution. ]


`/home/abrar/Research/stero_research_claude/model/scripts/sceneflow_loader.py`
[ SceneFlow Dataset Loader ]
[ PyTorch Dataset implementation for SceneFlow Driving, Monkaa, and FlyingThings3D. Reads images and .pfm disparity files, handling crops, normalization, and stereo specific augmentations. ]

`/home/abrar/Research/stero_research_claude/model/scripts/test_distilled_camera.py`
[ Distilled model camera test ]
[ Script to test the distilled StereoLite model on live camera feed. ]

`/home/abrar/Research/stero_research_claude/model/scripts/test_hitnet_camera.py`
[ HITNet camera test ]
[ Script to test the TinyHITNet baseline model on live camera feed. ]

`/home/abrar/Research/stero_research_claude/model/scripts/test_resolution_scaling.py`
[ Resolution scaling test ]
[ Tests how StereoLite handles various input resolutions dynamically. ]

`/home/abrar/Research/stero_research_claude/model/scripts/train_finetune_indoor.py`
[ Indoor finetuning script ]
[ PyTorch training script specifically for fine-tuning StereoLite on indoor datasets or real-world indoor camera captures. ]

`/home/abrar/Research/stero_research_claude/model/scripts/train_sceneflow.py`
[ SceneFlow training script ]
[ PyTorch training script for training StereoLite from scratch on the SceneFlow dataset. ]

`/home/abrar/Research/stero_research_claude/model/scripts/train_sharpness.py`
[ Sharpness fine-tuning script ]
[ PyTorch training script focused on fine-tuning for edge sharpness, possibly using specific loss variants like gradient/edge-aware loss. ]

`/home/abrar/Research/stero_research_claude/model/scripts/view_camera_feed.py`
[ Camera feed viewer ]
[ Opens CCB stereo camera and displays left/right feeds side by side. Used for testing camera hardware and frame rates (2560x720 at 60fps). ]

`/home/abrar/Research/stero_research_claude/model/scripts/watch_sharp_panels.py`
[ Training panels watcher ]
[ Watch a training-panel directory and display the latest step's panels in a cv2 window as they arrive. Useful for monitoring live training progress. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/techniques.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/design_lessons.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/SKILL.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/blocks.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/architectures.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/equations.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/stereo-vision-expert/papers_index.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/references/irb_decision_tree.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/references/preregistration_guide.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/references/socratic_questioning_framework.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/references/source_quality_hierarchy.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/references/mode_selection_guide.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/references/methodology_patterns.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/references/systematic_review_toolkit.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/references/logical_fallacies.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/.claude/skills/references/literature_monitoring_strategies.md`
[ Agent skill configuration ]
[ Claude agent instructions, references, and frameworks for executing automated research or diagram generation tasks. ]

`/home/abrar/Research/stero_research_claude/presentation/build_v2_deck.py`
[ Presentation deck asset ]
[ Scripts or markdown used to generate and compile the research presentation deck and associated figures. ]

`/home/abrar/Research/stero_research_claude/presentation/_md_to_pdf.py`
[ Presentation deck asset ]
[ Scripts or markdown used to generate and compile the research presentation deck and associated figures. ]

`/home/abrar/Research/stero_research_claude/presentation/build_v3_deck.py`
[ Presentation deck asset ]
[ Scripts or markdown used to generate and compile the research presentation deck and associated figures. ]

`/home/abrar/Research/stero_research_claude/presentation/build_v4_deck.py`
[ Presentation deck asset ]
[ Scripts or markdown used to generate and compile the research presentation deck and associated figures. ]

`/home/abrar/Research/stero_research_claude/presentation/build_v5_deck.py`
[ Presentation deck asset ]
[ Scripts or markdown used to generate and compile the research presentation deck and associated figures. ]

`/home/abrar/Research/stero_research_claude/presentation/figs/build_slide_figs.py`
[ Presentation deck asset ]
[ Scripts or markdown used to generate and compile the research presentation deck and associated figures. ]

`/home/abrar/Research/stero_research_claude/presentation/figs/build_progress_gifs.py`
[ Presentation deck asset ]
[ Scripts or markdown used to generate and compile the research presentation deck and associated figures. ]

`/home/abrar/Research/stero_research_claude/presentation/figs/build_intro_figure.py`
[ Presentation deck asset ]
[ Scripts or markdown used to generate and compile the research presentation deck and associated figures. ]

`/home/abrar/Research/stero_research_claude/presentation/figs/build_arch_diagrams.py`
[ Presentation deck asset ]
[ Scripts or markdown used to generate and compile the research presentation deck and associated figures. ]

`/home/abrar/Research/stero_research_claude/presentation/script/V5_Presentation_Script.md`
[ Presentation deck asset ]
[ Scripts or markdown used to generate and compile the research presentation deck and associated figures. ]

`/home/abrar/Research/stero_research_claude/papers/paper_registry.md`
[ Paper tracking and registries ]
[ Markdown file used for tracking paper metadata, performance benchmarks, and core concepts across the literature. ]

`/home/abrar/Research/stero_research_claude/papers/verified_performance.md`
[ Paper tracking and registries ]
[ Markdown file used for tracking paper metadata, performance benchmarks, and core concepts across the literature. ]

`/home/abrar/Research/stero_research_claude/papers/CONCEPTS.md`
[ Paper tracking and registries ]
[ Markdown file used for tracking paper metadata, performance benchmarks, and core concepts across the literature. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/transformer/NMRF.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/transformer/CroCov2.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/transformer/STTR.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/transformer/ELFNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/transformer/ChiTransformer.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/transformer/GOAT.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/transformer/_SYNTHESIS_transformer.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/transformer/GMStereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/transformer/CEST.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/transformer/BridgeDepth.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/datasets/Spring.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/datasets/ETH3D.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/datasets/DrivingStereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/datasets/Middlebury.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/datasets/SceneFlow.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/datasets/KITTI2015.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/datasets/_SYNTHESIS_datasets.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/datasets/KITTI2012.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/datasets/Booster.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/end_to_end/ACVNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/end_to_end/GCNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/end_to_end/GWCNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/end_to_end/PSMNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/end_to_end/_SYNTHESIS_end_to_end.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/end_to_end/MC-CNN.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/end_to_end/DispNetC.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/end_to_end/AANet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/end_to_end/CFNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/end_to_end/GANet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/nas/EASNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/nas/_SYNTHESIS_nas.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/nas/LEAStereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/iterative_variants/MC-Stereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/iterative_variants/StereoAnything.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/iterative_variants/GREAT-Stereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/iterative_variants/DLNR.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/iterative_variants/MoCha-Stereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/iterative_variants/Any-Stereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/iterative_variants/_SYNTHESIS_iterative_variants.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/iterative_variants/CREStereo++.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/iterative_variants/LoS.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier2/iterative_variants/ICGNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/foundation_model/PromptStereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/foundation_model/AIO-Stereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/foundation_model/FoundationStereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/foundation_model/MonSter.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/foundation_model/DEFOM-Stereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/foundation_model/Fast-FoundationStereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/foundation_model/StereoAnywhere.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/foundation_model/_SYNTHESIS_foundation_model.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/foundation_model/D-FUSE.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/Pip-Stereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/LightStereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/_SYNTHESIS_efficient.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/BANet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/GGEV.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/EdgeStereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/Distill-then-Prune.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/StereoDRNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/Separable-Stereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/LiteAnyStereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/CGI-Stereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/MADNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/FADNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/AutoDispNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/BGNet.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/HD3.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/CascadeCV.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/efficient/CoEx.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/iterative/CREStereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/iterative/RAFT-Stereo.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/iterative/_SYNTHESIS_iterative.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/surveys/_SYNTHESIS_surveys.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/surveys/Scharstein_Taxonomy_IJCV2002.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/surveys/Poggi_Synergies_TPAMI2021.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/surveys/Hirschmuller_SGM_TPAMI2007.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier1/surveys/Tosi_Survey_IJCV2025.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/_SYNTHESIS_tier3.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/transformer/FormerStereo_Zhang_ECCV2024.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/transformer/ViTAStereo_Zhang_TIV2024.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/confidence/SEDNet_Chen_CVPR2023.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/confidence/OnTheConfidence_Poggi_TPAMI2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/beyond_rgb/EventStereoSurvey_Ghosh_TPAMI2025.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/refinement/NDR_Aleotti_3DV2021.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/refinement/LaC_Liu_AAAI2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/refinement/SMD-Nets_Tosi_CVPR2021.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/refinement/ADL_Xu_CVPR2024.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/refinement/StereoRisk_Liu_ICML2024.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/HITNet_Tankovich_CVPR2021.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/MobileStereoNet_Shamsafar_WACV2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/PCVNet_Zeng_ICCV2023.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/AnyNet_Wang_ICRA2019.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/StereoNet_Khamis_ECCV2018.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/MABNet_Xing_ECCV2020.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/IINet_Li_AAAI2024.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/NVStereoNet_Smolyanskiy_CVPRW2018.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/DeepPruner_Duggal_ICCV2019.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/DecNet_Yao_CVPR2021.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/PBCStereo_Cai_ACCV2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/efficient/ADStereo_Wang_TIP2023.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/self_supervised/Reversing-Stereo_Aleotti_ECCV2020.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/self_supervised/NeRFStereo_Tosi_CVPR2023.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/self_supervised/MonoDepth_Godard_CVPR2017.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/datasets/TartanAir_Wang_IROS2020.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/datasets/WMGStereo_Yan_arXiv2025.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/PseudoLiDAR_Wang_CVPR2019.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/LIGAStereo_Guo_ICCV2021.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/S3M-Net_Wu_TIV2024.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/YOLOStereo3D_Liu_ICRA2021.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/DispSegNet_Zhang_RAL2019.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/SegStereo_Yang_ECCV2018.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/TaskPrompter_Ye_ICLR2023.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/DroNet_Loquercio_RAL2018.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/TwinLiteNet_Che_2023.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/RAFT-3D_Teed_CVPR2021.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/TiCoSS_Liu_2024.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/InvPT_Ye_ECCV2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/DSGN_Chen_CVPR2020.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/NDDR-CNN_Gao_CVPR2019.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/StereoRCNN_Li_CVPR2019.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/MultiTaskUncertainty_Kendall_CVPR2018.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/AurigaNet_2026.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/CrossStitch_Misra_CVPR2016.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/RealTimeSemStereo_Dovesi_2020.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/PLUMENet_Wang_IROS2021.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/SSPCV-Net_Wu_ICCV2019.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/MTI-Net_Vandenhende_ECCV2020.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/DSGNpp_Chen_2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/HybridNets_Vu_2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/YOLOP_Wu_MIR2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/PanopticPerceptionSurvey_2024.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/YoloSGN_Wang_UAV2025.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/MultiNet_Teichmann_IV2018.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/PADNet_Xu_CVPR2018.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/YOLOPv2_Han_2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/MSDESIS_Psychogyios_TMI2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/MTAN_Liu_CVPR2019.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/SemStereo_2024.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/MTLSurvey_Vandenhende_TPAMI2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/multi_task/SGNet_Chen_ACCV2020.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/domain_shift/GraftNet_Liu_CVPR2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/domain_shift/FCStereo_Zhang_CVPR2022.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/domain_shift/MRL-Stereo_Rao_CVPR2023.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/domain_shift/HVT_Chang_CVPR2023.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/domain_shift/DKT-Stereo_Zhang_CVPR2024.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/domain_shift/DSMNet_Zhang_ECCV2020.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/surveys/OpenStereo_Xianda_arXiv2023.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/papers/summaries/tier3/surveys/Laga_SurveyDeepStereo_TPAMI2020.md`
[ Paper summary ]
[ Markdown summary of a stereo vision research paper, extracting core contributions, architecture details, and results. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/download_sceneflow.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/download_eth3d.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/ablation_gce_in_tileinit.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/ablation_phase2_n100.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/ablation_phase3_hires.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/status_datasets.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/widener_apples_to_apples.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/seq_loss_gamma_sweep.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/eval_liteanystereo_middlebury2014.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/eval_middlebury2014.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/eval_igev_middlebury2014.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/ablation_expert_review.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/README.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/probe3_gpu.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/probe1_hello.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/ablation_baseline_n100.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/probe2_volume.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/ablation_pure_l1_diagnostic.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/ablation_phase3_compose.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/download_middlebury.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/demo_imgs.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/evaluate_stereo.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/save_disp.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/train_stereo.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/demo_video.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/core/extractor.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/core/__init__.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/core/geometry.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/core/igev_stereo.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/core/submodule.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/core/stereo_datasets.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/core/update.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/core/utils/__init__.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/core/utils/utils.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/core/utils/frame_utils.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/igev_stereo_repo/core/utils/augmentor.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/profile_speed.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/flops_count.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/evaluate_stereo.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/README.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/Utils.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/demo.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/core/__init__.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/core/fnet.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/core/liteanystereo.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/core/submodule.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/core/stereo_datasets.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/core/aggregation.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/core/utils/__init__.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/core/utils/utils.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/scripts/modal/lite_any_stereo_repo/core/utils/frame_utils.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/widener_modal_20260502-083147/REPORT.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/widener_modal_20260502-083147/widener_modal_run/costlookup_yolo26n_full_f2_only/README.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/widener_modal_20260502-083147/widener_modal_run/costlookup_yolo26n_full_dw/README.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/widener_modal_20260502-083147/widener_modal_run/costlookup_yolo26s_full_native/README.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/widener_modal_20260502-083147/widener_modal_run/costlookup_yolo26n_full_none/README.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/widener_modal_20260502-083147/widener_modal_run/costlookup_yolo26n_full_ghostconv/README.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/widener_modal_20260502-083147/widener_modal_run/tilegru_yolo26n_p1/README.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/widener_modal_20260502-083147/widener_modal_run/costlookup_yolo26n_full_topdown_fpn/README.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/widener_modal_20260502-083147/widener_modal_run/costlookup_yolo26n_full_f2_f4/README.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/benchmarks/widener_modal_20260502-083147/widener_modal_run/costlookup_ghost_p1/README.md`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/training/losses.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/training/__init__.py`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite_yolo/stereolite_architecture_doc.tex`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/model/designs/StereoLite/stereolite_architecture_doc.tex`
[ Project file ]
[ Project source file or script for modeling, data handling, or visualization. ]

`/home/abrar/Research/stero_research_claude/review_paper/main.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/sections/04_compression_taxonomy.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/sections/08_conclusion.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/sections/07_roadmap.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/sections/02_background.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/sections/06_edge_hardware.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/sections/01_introduction.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/sections/05_generalization.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/sections/03_foundation_baseline.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/sections/_tables/tab_compression_families.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/sections/_tables/tab_efficient_comparison.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/sections/_tables/tab_datasets.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/figures/fig_taxonomy_tikz.tex`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/figures/_data/make_timeline.py`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/figures/_data/make_family_contribution.py`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/figures/_data/make_taxonomy.py`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/figures/_data/make_pareto.py`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/figures/_data/method_data.py`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

`/home/abrar/Research/stero_research_claude/review_paper/figures/_data/make_param_pareto.py`
[ LaTeX review paper source ]
[ LaTeX source files, sections, or styles for compiling the comprehensive IEEE review paper on stereo vision. ]

