# Target Journal Selection, Author Guidelines, and Submission Readiness Assessment

## 1. Executive Summary of Journal Selection

Following a rigorous 15-criterion evaluation of five peer-reviewed, Scopus- and Web of Science (SCIE)-indexed journals in biomedical engineering and medical machine learning (documented in `journal_comparison.xlsx`), we select **MDPI *Bioengineering*** (ISSN: 2306-5354; SCIE Impact Factor: 3.8; Scopus CiteScore: 4.8, Q2 *Engineering, Biomedical*) as our **Primary Target Journal**, with **Elsevier *Biomedical Signal Processing and Control*** (ISSN: 1746-8094; SCIE Impact Factor: 4.9; Scopus CiteScore: 8.2, Q1 *Biomedical Engineering*) designated as the **#1 Zero-APC Hybrid Alternative**.

Both journals use the identical **numbered square-bracket `[1]` reference format** in order of appearance and standard IMRaD biomedical engineering structure, enabling seamless submission to either venue without structural reformatting.

---

## 2. Rationale for Selecting *Bioengineering* (MDPI)

1. **Direct Topical Continuity with Focal Literature**:
   - Our primary benchmark directly addresses the methodological gaps identified by **Rakotomanana & Rouhafzay (2025)** (*"A Scoping Review of AI-Based Approaches for Detecting Autism Traits Using Voice and Behavioral Data"*, *Bioengineering*, 12(11), 1136, DOI: `10.3390/bioengineering12111136`).
   - Furthermore, **Pradhan, Chester, & Padhiar (2022)** (*"Classification of Autism and Control Gait in Children Using Multisegment Foot Kinematic Features"*, *Bioengineering*, 9(10), 552, DOI: `10.3390/bioengineering9100552`) established the precedent in *Bioengineering* for combining pediatric ASD gait kinematics with SHAP explainability.
2. **Verified Indexing**:
   - **Scopus**: Indexed (CiteScore 2024: 4.8; SNIP: 0.96).
   - **Web of Science**: Science Citation Index Expanded (SCIE; Impact Factor: 3.8, Q2 in *Engineering, Biomedical*).
   - **PubMed / PubMed Central (PMC)**: Fully indexed in PMC and MEDLINE/PubMed.
3. **Article Type & Length Compatibility**:
   - Accepted as a full-length **Original Research Article**.
   - Supports comprehensive methodological exposition, multi-regime validation tables, mathematical normalization derivations, and 10 high-resolution (300 DPI) figures without restrictive page-count truncation.

---

## 3. Author Guideline Compliance Checklist

| Requirement Category | Target Journal Specification (*Bioengineering* / *BSPC*) | Manuscript Implementation Status |
| :--- | :--- | :--- |
| **Article Title** | Concise, informative, avoiding abbreviations where possible | *Movement-Based Classification of Autism Spectrum Disorder in Children Using CatBoost: A Methodologically Rigorous Benchmark* (**Compliant**) |
| **Structured Abstract** | ~200–250 words covering Background/Objectives, Methods, Results, and Conclusions | 245-word structured abstract reporting exact sample sizes ($N=100$ nominal, $N=88$ unique), all 4 validation regimes, and top TreeSHAP biomarkers (**Compliant**) |
| **Keywords** | 5–8 specific indexing terms | Autism Spectrum Disorder (ASD); Kinect v2 3D skeleton; gait kinematics; CatBoost; data leakage; StratifiedGroupKFold; TreeSHAP; explainable AI (**Compliant**) |
| **Highlights (for Elsevier BSPC)** | 3–5 bullet points ($\le 85$ characters each including spaces) | Included in front matter (**Compliant**) |
| **Main Text Structure** | 1. Introduction, 2. Related Work, 3. Dataset & Forensic Audit, 4. Methodology, 5. Experimental Setup, 6. Results, 7. Explainability & Clinical Biomechanics, 8. Discussion, 9. Limitations, 10. Conclusion | Complete 10-section IMRaD+ architecture with all 21 required subsections (**Compliant**) |
| **Reference Style** | Numbered square brackets `[1]` in order of appearance; full author lists, titles, journal names, volume, issue, pages/article ID, and DOI | 26/26 references verified against Crossref/PubMed with active DOIs; 100% bidirectional in-text $\leftrightarrow$ bibliography correspondence (**Compliant**) |
| **Figures & Tables** | Minimum 300 DPI resolution, self-contained captions, multi-panel labels `(a)`, `(b)`, `(c)`, no overlapping text | 10 original 300-DPI figures (`fig01`–`fig10`) and 8 comprehensive statistical tables embedded and cross-referenced (**Compliant**) |
| **Ethics Statement** | Secondary analysis of publicly archived, de-identified human movement dataset (`10.5061/dryad.s7h44j150`) with original IRB/consent declaration | Included in Back Matter (**Compliant**) |
| **Data & Code Availability** | Direct repository path, Dryad DOI, preprocessing/feature scripts, random seeds (`42`), and exact hyperparameter tables | Included in Back Matter and `03_methodology/reproducibility_guide.md` (**Compliant**) |
| **AI Disclosure Statement** | Explicit disclosure of AI-assisted code auditing, literature verification, and manuscript preparation with human/execution verification | Included in Back Matter and `07_quality_audit/similarity_risk_audit.md` (**Compliant**) |

---

## 4. Submission Readiness Assessment

- **Scientific Defensibility**: **Ready for Submission**. Unlike prior studies on the Al-Jubouri (2020) Kinect v2 dataset that either suffered from trial-level augmentation leakage [3] or overlooked the 12 cross-folder duplicate subjects [4–7], this manuscript provides a complete forensic audit, eliminates the $+16.6\text{ cm}$ stature confound via per-frame 3D torso normalization, and reports performance transparently across all four validation regimes (`R1` Naive, `R2` Nominal 100-Subject SGKF, `R3` Cluster-Aware/Deduplicated 88-Subject SGKF, and `R4` Unaugmented Original-Only).
- **Formatting & File Completeness**: **Ready for Submission**. Editable Markdown source (`editable_source.md`), Microsoft Word manuscript with embedded 300-DPI figures and formatted tables (`final_manuscript.docx`), and typeset PDF (`final_manuscript.pdf`) are generated and verified in `research_output/01_manuscript/`.
