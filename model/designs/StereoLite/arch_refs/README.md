# TIER A architecture-diagram references for StereoLite v9

Copies of the best stereo architecture figures from our corpus. Curated to inform
the StereoLite v9 diagram. Ranked roughly best-to-good within the tier.

## TIER A (17)

1. **RAFTStereo_fig1_architecture.png** - canonical clean template.
   Input thumbnails on the left, two blue encoder stacks (3D isometric conv
   prisms), a dashed "correlation pyramid" inset floating above the main row,
   a row of four green GRU cells each tapped by a yellow "L" lookup, and a
   disparity thumbnail on the right. Three colours total (blue encoder, green
   recurrent, orange pyramid); flow is strictly left-to-right; the inset keeps
   the pyramid from cluttering the main row. Nothing decorative.

2. **Tosi2025_fig2_RAFT_architecture.png** - same as above, redrawn cleaner
   for the survey. Even tighter spacing, labels directly under the object.
   Worth studying as a clean re-render of the RAFT canon.

3. **DEFOM_fig3_architecture.png** - best legend+callout discipline in the
   corpus. Uses a boxed top-right legend explaining SL/PL lookups and
   operators. Colours: blue feature, yellow context, green depth-foundation.
   The "Scale Update" and "Delta Update" stages are separated by vertical
   dashed grouping rails labelled at the bottom. Shows modulated-depth
   thumbnail in-line so the reader immediately sees what the DAv2 branch
   produces.

4. **IGEV_fig3_architecture.png** - clean bottom-left legend (2D conv / 3D
   conv / ConvGRU as three coloured bars), side-by-side main and inset panels,
   and every cube annotated with its resolution fraction. Weakness: the 3D
   regularisation network is a little cramped.

5. **IGEVpp_fig3_architecture.png** - extends IGEV with multi-scale cost
   volumes. Demonstrates how to lay out three parallel cost-volume stacks
   (small/medium/large disparity) without confusion: horizontal rails with
   a single fusion node on the right. Good blueprint for our "cost volume at
   1/16 but cascade upsamples" story.

6. **Selective_fig2_overview.png** - multi-level attention map drawn
   below the main flow as an auxiliary panel, not as a colour overlay on the
   main row. This is the correct way to show a secondary output without
   overloading the primary figure.

7. **StereoAnywhere_fig2_architecture.png** - heavy but well-organised. Uses
   circled step numbers (1)-(6) along the data path so the caption can walk
   through the pipeline in order. Great trick for a dense architecture.

8. **FoundationStereo_fig2_architecture.png** - dual-branch (mono + stereo)
   done right. A single large AHCF inset on the right shows the transformer
   detail while the main row stays legible. The position-encoding and
   element-wise-product operators have their own legend icons at the bottom.

9. **FastFS_fig3_architecture.png** - three stacked horizontal panels each
   labelled with the paper section (S3.1, S3.2, S3.3). Demonstrates how to
   present a three-stage pipeline where each stage has its own sub-figure
   without making the whole figure vertical. Light pastel backgrounds behind
   each panel.

10. **GGEV_fig2_architecture.png** - compact and tight. Everything in one
    row, three colour families (encoder, cost-volume, GRU), and the iterative
    updates shown as a vertical stack at the right with "dot dot dot" between
    them to indicate repetition. Good model for showing many GRU iterations
    without drawing them all.

11. **PromptStereo_fig2_overview.png** - very clean rectangular block style.
    Each block is a labelled outlined rectangle, not an isometric prism.
    Shows that 2D block diagrams can be as readable as isometric ones if
    spacing is honest. A separate "Freeze" icon in the legend indicates which
    modules are fine-tuned; useful if we show distillation.

12. **T1_CGI-Stereo_fig2_architecture.png** - best inline sub-module
    expansion. The CGF module is drawn full-detail as a side-panel with a
    thin dashed line linking it to the block in the main flow that it belongs
    to. No callout lines crossing the main flow.

13. **T1_CoEx_fig2_architecture.png** - concise one-line flow, with a small
    boxed inset of the Guided Channel Excitation equation-as-diagram. Shows
    how a single small detail box can explain a mechanism without an entire
    side panel.

14. **T2_PSMNet_arch.png** - still the reference for hourglass-plus-cost-volume.
    Spatial Pyramid Pooling expanded as a dashed-red inset on the left; stacked
    hourglass expanded as a dashed-yellow inset on the right; main row in the
    middle. A masterclass in using two symmetrical insets to declutter.

15. **T2_GANet_arch.png** - shows SGA and LGA sub-modules as labelled
    sub-panels (b) and (c) below the main arch panel (a). Worth copying this
    "one main + two sub-panels" composition if we want to highlight plane
    propagation and convex upsample as sub-modules of the v9 pipeline.

16. **T2_NMRF_arch.png** - not an architecture in the visual sense but
    demonstrates how to show a comparison row (left vs ours) with error-map
    ribbons rather than burying it in the text. Useful if we ever want an
    auxiliary "what the model produces" strip.

17. **BANet_fig2_architecture.png** - good example of a dual-branch aggregator
    (smooth vs detailed) drawn as two parallel rails. Clean colour coding
    (pink for scale-attention, two cubes with different opacities for the
    two cost volumes).

## TIER B (competent, not exceptional)

AANet, ACVNet, GWCNet, AIO, CFNet, BGNet, CREStereo_fig2, CascadeCV, StereoDRNet,
DLNR, LoS, MADNet, MoCha, AnyStereo, Bridge, GREAT, MCStereo, GMStereo, GOAT,
ELF, EAS, ICGNet, MonSter. These are legible and informative but suffer from one
or more of: cramped labels, too many colours without a legend, decorative
isometric 3D rendering, or a main row that would need a caption walkthrough to
follow.

## TIER C (cluttered, outdated, or unclear)

- **HD3_fig2_architecture.png** - pure abstract symbol soup (math blocks and
  circles), no ground-truth image or disparity, reader has no visual anchor.
- **FADNet_fig1_architecture.png** - two stacked U-Nets rendered with 40+
  identical blue prisms in a single row, visually indistinguishable blocks.
- **EdgeStereo_fig1_architecture.png** - too many arrows crossing each
  other, labels fighting for space, overall dense and hard to parse.
- **MADNet_fig1_architecture.png** - triangular "pyramid tower" metaphor
  that does not line up with data flow; unusual layout hurts readability.
- **AutoDispNet_fig1/fig2** - NAS cell diagrams, useful only for NAS papers.
- **CSTR** - three-letter acronyms crammed next to one another, no legend.
- **MCCNN_arch.png** - 2015-era block diagram, text-only, no visual element.
- **DispNetC_arch.png** - a table, not a figure.
- **StereoAnything_arch.png** - a text-only page, wrong file.
- **CREStereo_fig4_search_window.png** - a schematic detail, not an arch fig.
- **CroCo_arch.png** - dominated by body text; the actual architecture panel
  is a thumbnail in the corner.
- **ChiTransformer, GCNet** - tiny all-blue diagrams, no functional colour.
