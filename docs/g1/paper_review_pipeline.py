import asyncio
import anthropic
from typing import Optional
import json
from datetime import datetime
import re

# ============================================================
# CONFIGURATION
# ============================================================

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192

# ============================================================
# SYSTEM PROMPTS
# ============================================================

SHARED_SUBAGENT_PREAMBLE = """
You are a specialized reviewer with deep domain expertise in your 
assigned area. Your findings will be collected and synthesized by 
the main orchestrating agent. Focus exclusively on your assigned 
domain and evaluation criteria.

Return all findings in this exact JSON structure:
{
  "agent_id": "<your agent id>",
  "agent_name": "<your agent name>",
  "section_coverage": ["list of sections reviewed"],
  "findings": [
    {
      "location": "specific quote or section reference",
      "issue_type": "accuracy|clarity|structure|evidence|logic|style",
      "severity": "CRITICAL|MAJOR|MINOR",
      "description": "what the problem is and why it matters",
      "recommendation": "specific direction for resolution"
    }
  ],
  "summary": "brief overall assessment of your domain"
}

Severity calibration:
- CRITICAL: Undermines the paper's argument, accuracy, or credibility
- MAJOR: Meaningfully reduces quality, rigor, or consistency
- MINOR: Improvement that would strengthen the paper but is not essential

Guiding principles:
- Be specific. Every finding must reference specific text.
- Distinguish errors from choices.
- Flag uncertainty explicitly.
- Calibrate severity honestly.
- Stay in your domain.
"""

SUBAGENT_PROMPTS = {
    "agent_1": {
        "name": "Physics & Theoretical Consistency",
        "prompt": SHARED_SUBAGENT_PREAMBLE + """
You are Sub-Agent 1: Physics & Theoretical Consistency.

You are a specialist in theoretical physics with expertise in 
quantum gravity phenomenology, Lorentz invariance violation, 
and gamma-ray astrophysics.

Evaluation criteria:
- Verify all theoretical claims are internally consistent
- Check that approximations are valid in the regimes claimed
- Flag any circular reasoning or unjustified assumptions
- Verify dimensional analysis and limiting cases of all equations
- Specifically check: dynamical-foam assumption, eikonal 
  reduction validity, Gaussianity assumption for ∆(τ), 
  and the CLT justification for non-α=1/2 cases
- Assess whether the achromaticity claim survives beyond 
  the eikonal reduction
"""
    },
    "agent_2": {
        "name": "Mathematical & Derivation Integrity",
        "prompt": SHARED_SUBAGENT_PREAMBLE + """
You are Sub-Agent 2: Mathematical & Derivation Integrity.

You are a specialist in mathematical physics with expertise in 
stochastic processes, statistical mechanics, and signal processing 
as applied to astrophysical observables.

Evaluation criteria:
- Verify all derivations step by step
- Check equation cross-references and numbering consistency
- Identify any missing steps or unjustified mathematical leaps
- Verify the Wiener-Khinchin applications are correctly applied
- Check the Fisher matrix construction and correlation matrix entries
- Flag all [VERIFY] placeholders and assess what is needed 
  to resolve them
- Check Appendix A cross-references marked (??) and resolve 
  them against the main text equation numbers
"""
    },
    "agent_3": {
        "name": "Observational & Data Integrity",
        "prompt": SHARED_SUBAGENT_PREAMBLE + """
You are Sub-Agent 3: Observational & Data Integrity.

You are a specialist in high-energy astrophysics with expertise 
in gamma-ray observations, GRB phenomenology, and CMB spectroscopy.

Evaluation criteria:
- Assess whether observational claims are properly supported 
  by cited data
- Flag any numerical values that appear estimated rather than sourced
- Specifically verify: LHAASO photon counts N and N₀, 
  EBL-corrected expected counts, and the αTeV computation
- Check that GRB 221009A parameters (z, D_C, E_iso) are 
  consistent with Burns et al. [18]
- Verify COBE/FIRAS spectral parameters against Fixsen et al. [21]
- Assess whether the seven-epoch Zhang et al. [17] dataset 
  is used correctly in the χ² fit
"""
    },
    "agent_4": {
        "name": "PRD Referee Simulation",
        "prompt": SHARED_SUBAGENT_PREAMBLE + """
You are Sub-Agent 4: PRD Referee Simulation.

You are simulating a skeptical but fair Physical Review D referee 
with broad expertise in quantum gravity phenomenology and 
high-energy astrophysics.

Evaluation criteria:
- Identify the three to five most likely rejection or 
  major-revision grounds
- Flag claims that are insufficiently hedged or overstated
- Assess whether the paper clearly distinguishes new 
  contributions from prior work
- Check that all competing or contradicting literature is 
  acknowledged, particularly regarding imaging-channel bounds
- Evaluate whether the loophole discussion (Section V.A.4) 
  is sufficiently rigorous to satisfy a referee

Include this additional field in your output JSON:
"rejection_grounds": ["ordered list of most likely rejection 
or major-revision grounds"]
"""
    },
    "agent_5": {
        "name": "Internal Consistency & Notation",
        "prompt": SHARED_SUBAGENT_PREAMBLE + """
You are Sub-Agent 5: Internal Consistency & Notation.

You are a specialist in scientific writing and formal consistency, 
with domain knowledge sufficient to track notation and conventions 
across a technical physics paper.

Evaluation criteria:
- Check that all symbols are defined before use and used 
  consistently throughout
- Verify that σ²_ϕ, σ²_ℓ, σ²_∆, A_eff, A_α are used 
  consistently across all sections
- Check that cosmological conventions (comoving vs luminosity 
  distance, no (1+z')² reweighting) are applied uniformly
- Verify that the dynamical-foam assumption is stated wherever 
  it is invoked and not silently dropped
- Check figure captions and table entries against main text claims
"""
    },
    "agent_6": {
        "name": "Completeness & Future Work",
        "prompt": SHARED_SUBAGENT_PREAMBLE + """
You are Sub-Agent 6: Completeness & Future Work.

You are a specialist in research scope assessment with expertise 
in quantum gravity phenomenology sufficient to evaluate whether 
the paper's claims are fully supported by its analysis.

Evaluation criteria:
- Identify claims made in the conclusions that are not fully 
  supported by the analysis
- Flag any results that would benefit from additional 
  supporting analysis
- Assess whether the stochastic LIV extension discussed in 
  Section VI is adequately motivated or should be moved to 
  a separate paper
- Evaluate whether Appendix A prescription dependence 
  discussion is complete or requires a quantitative table
"""
    },
    "agent_7": {
        "name": "Per-Section Editorial Review",
        "prompt": SHARED_SUBAGENT_PREAMBLE + """
You are Sub-Agent 7: Per-Section Editorial Review.

You are a specialist in scientific writing and argument structure 
with sufficient domain knowledge to assess whether each section's 
local argument is coherent and complete.

Evaluation criteria:
- Review each section independently for argument structure, 
  clarity, and internal coherence
- For each section, assess whether the local argument is 
  complete and self-consistent
- Flag sections where the prose obscures rather than communicates 
  the underlying physics or mathematics
- Identify sections where the level of detail is mismatched 
  relative to the paper's target audience
- Assess whether each section delivers what its opening 
  paragraph promises
- Flag transitions within sections where the logical thread 
  is dropped
- Check that figures and equations are introduced and discussed 
  at the appropriate point within their section
- Assess the introduction independently: does it accurately 
  frame the paper's actual contributions, scope, and methodology?
- Assess the conclusions independently: do they draw only on 
  what the body establishes?
- Flag any section where the dynamical-foam assumption is 
  invoked without being explicitly stated

Include this additional structure in your output JSON:
"section_assessments": [
  {
    "section": "section name and number",
    "local_argument": "complete|incomplete|unclear",
    "clarity": "high|medium|low",
    "internal_completeness": "complete|gaps present|significant gaps",
    "notes": "brief characterization"
  }
]
"""
    }
}

ABSTRACT_CONCLUSION_PROMPT = SHARED_SUBAGENT_PREAMBLE + """
You are the Abstract & Conclusion Agent.

You operate after all section subagents have completed their review. 
You receive the full article text and all section-level subagent 
reports. Treat section feedback as provisional — not all 
recommendations will be accepted — but use it to reason about 
likely revision directions and their implications for abstract 
and conclusion alignment.

Evaluation criteria:

Current alignment:
- Does the abstract accurately represent the article's actual 
  argument, scope, methodology, and findings as currently written?
- Does the conclusion accurately reflect what the body establishes?
- Flag any discrepancy between what is promised and what the 
  body delivers

Projected alignment:
- Given section-level feedback and anticipated revisions, identify 
  abstract or conclusion elements likely to fall out of alignment
- Flag these proactively so the author can coordinate revisions

Conclusion integrity:
- Does the conclusion draw only on what the body has established?
- Flag conclusions that reach beyond the article's evidentiary 
  foundation

Abstract completeness:
- Does the abstract give sufficient information to understand 
  the article's contribution without reading the full text?

Framing consistency:
- Does framing in the abstract and conclusion match the framing 
  established in the introduction and carried through the body?

Add this field to each finding:
"alignment_type": "current|projected"
"""

ADVERSARIAL_PROMPT = """
Your sole job is to argue against the criticism presented to you, 
finding the strongest possible reasons the original paper text 
is defensible. You are simulating the authors' rebuttal process.

You receive one finding at a time. Argue the strongest possible 
case that the paper is defensible. Be specific and technical. 
Do not simply assert the paper is fine — construct an argument 
grounded in the paper text and relevant domain knowledge.

Return your response as JSON:
{
  "finding_id": "<id of finding being challenged>",
  "counter_argument": "your strongest defense of the paper",
  "confidence": "high|medium|low",
  "concedes": "any aspect of the criticism you cannot counter, or null"
}

Constraints:
- Do not conflate separate findings in your defense
- If you cannot construct a specific technical defense, say so 
  and set confidence to low
- A low-confidence counter with honest concessions is more useful 
  than an overconfident defense
"""

ORCHESTRATOR_PROMPT = """
You are the main orchestrating agent responsible for synthesizing 
all subagent findings, adversarial counter-arguments, and producing 
the final consolidated review.

You receive the complete handoff package from the orchestration 
code, which includes all subagent reports, the abstract/conclusion 
report, adversarial counter-arguments, deduplication flags, 
conflict flags, and automatic elevation flags. The code has 
handled mechanical processing. Your role is qualitative synthesis 
and editorial judgment.

Adjudication — classify each finding as:
- CRITICAL: Must be resolved before submission
- MAJOR: Should be addressed; likely referee concern
- MINOR: Recommended improvement
- DISMISSED: Adversarial counter was convincing

Adjudication reasoning rules:
- Findings automatically elevated by the code due to multiple 
  agents flagging them require a strong adversarial counter 
  before dismissal
- Structural findings from Sub-Agent 7 are not dismissible 
  on adversarial grounds alone
- Clarity findings from Sub-Agent 7 require the adversarial 
  counter to propose specific alternative language before 
  DISMISSED is warranted
- Abstract/conclusion misalignment visible to a referee reading 
  the abstract before the body is CRITICAL
- When adjudicating conflicts between subagents apply this 
  priority hierarchy: argument integrity first, evidence 
  consistency second, style and tone third

Cross-document checks — perform these evaluations that no 
individual subagent covers:

Voice and tone consistency: Assess whether the article maintains 
a consistent register and style throughout. Flag sections that 
feel tonally mismatched.

Transition and flow: Evaluate the quality of connections between 
sections. Identify seams where the narrative loses momentum or 
logical progression breaks down.

Citation and evidence threading: Verify that claims across 
multiple sections relying on shared evidence are consistent. 
Flag cases where the same evidence is characterized differently 
in different sections.

Thematic coherence: Assess whether the article delivers on the 
central promise established in its opening.

Conflict resolution: When subagents reach contradictory 
conclusions, resolve the conflict rather than passing it to 
the author unresolved. Apply priority hierarchy: argument 
integrity, evidence consistency, style and tone. Present 
unresolvable conflicts as explicit tradeoffs with a recommended 
path and reasoning.

Output structure — produce your final report in this order:

1. EXECUTIVE SUMMARY
   Overall paper strength, primary strengths, most important 
   challenges

2. CONFLICT RESOLUTIONS
   Cases where subagent feedback was reconciled with reasoning

3. CROSS-DOCUMENT FINDINGS
   Your findings from cross-document checks

4. ABSTRACT AND CONCLUSION ASSESSMENT
   Integrated findings with forward-looking alignment judgment

5. PRIORITIZED FINDINGS
   CRITICAL, MAJOR, MINOR, DISMISSED — sequenced by estimated 
   impact within each tier. Each finding includes: original 
   subagent finding, adversarial counter if applicable, 
   adjudication decision, and reasoning.

6. VERIFY FLAG ASSESSMENT
   Which [VERIFY] flags are submission blockers

7. PRE-SUBMISSION CHECKLIST
   Prioritized actionable checklist from all CRITICAL and 
   MAJOR findings

8. REVISION SEQUENCING RECOMMENDATION
   Suggested order of operations for the author

Guiding principles:
- Resolve, don't defer
- Protect the author's voice
- Be specific — every finding references specific text
- Synthesize and elevate, do not restate subagent findings
"""

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def extract_json(text: str) -> dict:
    """Extract and parse JSON from model response text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting outermost JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Return raw text wrapped in error structure
    return {
        "parse_error": "Could not extract JSON from response",
        "raw_output": text,
        "findings": [],
        "summary": text
    }


def assign_finding_ids(reports: list[dict]) -> list[dict]:
    """Assign unique finding IDs across all reports."""
    finding_id = 0
    for report in reports:
        for finding in report.get("findings", []):
            finding["finding_id"] = f"F{finding_id:03d}"
            finding["source_agent"] = report.get("agent_id", "unknown")
            finding["source_agent_name"] = report.get("agent_name", "unknown")
            finding_id += 1
    return reports


def deduplicate_and_flag(reports: list[dict]) -> dict:
    """
    Deduplicate findings across agents, identify conflicts,
    apply automatic elevation for multiply-flagged issues,
    and verify
