---
title: A brain-wide atlas of intrinsic neural timescales in mice
abstract: |
    This is a 100-150 word summary of our research, including the main objective, methods, key results, and conclusions. The abstract should provide readers with a clear overview of what the micropublication contains and its significance. Include the research question or hypothesis, the methodology employed, the key findings, and the main conclusions or implications of the work. This summary helps readers quickly assess whether the full content is relevant to their interests.

acknowledgments: |
    This work was supported by the Impact Scholars Program. We acknowledge the contributions of [former team members, teaching assistants, or mentors whose involvement does not meet the criteria of any authorship role].We thank the International Brain Laboratory for the publicly available BrainWideMap dataset.
---

# Description

In this section, we elaborate on our single claim, observation, or method being presented. Depending on the length of our Abstract, this section contains a maximum of 1350-1400 words.

We begin by providing context and background information, explaining why the research is important and what it aims to achieve. In the text, we remember to add citations to refer to the work of others, ensuring that proper credit is given and that the research is situated within the existing body of knowledge @pollockPracticalGuidanceCrafting2023. We do this using a bibliography file (`references.bib`), or linking directly to a DOI.

We continue by outlining our research approach, mentioning the datasets, tools, theoretical frameworks, and analytical methods we have used. This provides readers with the necessary information to understand and potentially replicate our work.

We then describe our results clearly, concisely, and in a logical order. We use a single high-resolution figure to support the findings and reference it in the text (@figure-main A). #@name panel_alphabet


# Math_Expressions

$ \tau_{\mathrm{eff}} $
\
$p < 10^{-26}$
\
$M = 1\text{–}4$
\
$r = 0.78$
\
$\text{slope} = 0.69$
\
for units, $\text{ms}$


# Introduction
The functional specialization of brain regions is deeply intertwined with their temporal dynamics, specifically the timescale over which local circuits maintain persistent activity. Quantified as the decay time constant (τ) of spontaneous spiking autocorrelation, intrinsic neural timescales (ITs) have emerged as a fundamental metric of this process (Murray et al., 2014). Longer timescales have been linked to cognitive computations that require information to be held over time, including working memory, evidence accumulation, and decision making (Cavanagh et al., 2016; Wasmuht et al., 2018; Bernacchia et al., 2011). Yet the mechanisms by which these dynamics arise, and how they are organized across the full brain, remain poorly understood. Answering these questions requires moving beyond isolated cortical recordings toward a brain-wide characterisation of temporal integration.
However, constructing this global map presents significant methodological challenges. Classical binned autocorrelation functions (ACFs) systematically underestimate timescales for neurons with low firing rates or bursty dynamics (Pochinok et al., 2026), conditions common across the mouse brain. Furthermore, fitting a single exponential decay assumes each neuron operates on only one characteristic timescale. To address both limitations, we apply the intrinsic Spike Time Tiling Coefficient, an unbinned iSTTC estimator (Pochinok et al., 2026) alongside Bayesian Information Criterion (BIC)-based multi-exponential modeling. This combination was critical not only for accurately characterising timescales in regions with extreme firing rate dynamic, but also for resolving the multi-component autocorrelation structure present in 73.9% of analyzed units (@fig-s1), revealing a coupled fast ($ \tau_{\mathrm{1}} $) and slow ($ \tau_{\mathrm{2}} $) dynamical architecture that single-exponential methods would obscure entirely, enabling a more accurate and complete characterisation of the brain-wide temporal hierarchy.

# Methods
# Results
# Discussion


```{figure} figure.png
:name: figure-main
:alt: Multi-panel figure supporting the main findings

\
**A.** Here we describe panel A.
\
**B.** Here we describe panel B.
\
**C.** Here we describe panel C.
```

Finally, we interpret our results, discussing their implications and relevance to the field. We provide a clear takeaway message for the reader that summarizes the contribution of this micropublication.


# Supplementary

## Supplementary Figure S1




```{figure} ./figures/cosmos_its.png
:kind: supplementary figure
:name: fig-s1
:alt: Supplementary figure S1

**Supplementary Figure 1**

**Distribution of the number of timescale components per neuron.** The majority of well-fitted neurons were best described by two-timescale models (60.4%), followed by one-timescale (25.7%) and three-timescale (13.8%) models, with only 0.1% requiring four timescales. The optimal number of components was selected using the Bayesian information criterion (BIC), with the constraint that each component contributed at least 1% to the overall autocorrelation shape. 

