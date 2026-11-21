# Beating Substrate Theory

This repository accompanies the English and French manuscripts on Beating Substrate Theory (BST). It contains the model inputs, scientific programs, Lean sources and offline observatory used to examine the results. The introduction links to three reading paths: physics, neuroscience and cognition, and philosophy. Chapter 2 follows one example from its equation of motion to a measurable signal. Chapter 6 can be read independently and begins with evidence from living systems. Appendix A gives the result and scope of each requirement.

| Resource | Location |
| --- | --- |
| Manuscript, English | [PDF](paper/manuscript-en.pdf) · [editable text](paper/manuscript-en.md) |
| Manuscrit, français | [PDF](paper/manuscript-fr.pdf) · [texte modifiable](paper/manuscript-fr.md) |
| Interactive projections and GIF export | [Open the observatory](observatory/index.html) |
| Prepared movies | [Reference films and examples](examples/) · [118 compact neutral films](examples/neutral-atlas/) |
| Claims, conditions and replay sections | [Requirement register](evidence/requirements.csv) · [chapter guide](evidence/claim-guide.csv) · [output file index](evidence/reader-files.csv) |
| Primary sources and external data | [Bibliography](evidence/bibliography.csv) · Appendix D |
| Numerical implementation and formal statements | [Scientific sources](reference/sources/) · [Lean projects](proofs/) |
| Verification performed for this edition | [Verification record](evidence/verification.md) |

## Reproduce a result

Python 3.12 on a POSIX system was used for verification. Run these commands from the repository root. The integrity check needs only Python. Each replay writes into a new or empty directory. You can keep the distributed inputs and your outputs side by side.

```sh
python3 replay.py integrity
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python replay.py pair --output ../bst-pair
python replay.py list
```

The pair example integrates an internal beat coupled to the distance between two centres. Its output contains the trajectory, a plot and a JSON report. The `reduced-dynamics` section checks the time step and what information is lost when the internal state is omitted. In the measurements tab, the default paired-wave trace reproduces the first 20 time units of this trajectory. The full numerical example extends to 160.

| Question | Section after `python replay.py` |
| --- | --- |
| Does the reduced pair dynamics retain information about phase? | `pair`, `reduced-dynamics` |
| When do a propagating signal and a moving clock define the same cone? How do the atomic inputs compare? | `substrate`, `clocks-and-flux`, `atomic-modes` |
| Which observables are evaluated in the eleven domains? | `domain-observables`, `physics` |
| How were the early controls and phase closures constructed? | `foundations` |
| What does the paired-history test establish about persistence, readout and agency? | `cognition`, `memory-extension`, `proofs` |
| Can the cosmological transfer and band scores be reconstructed? | `cosmology` |
| Which records are present for a given element? | `atlas --element Xe` |
| How does a new neon observation compare with the frozen predictions? | `neon --pressure 3.1 --interval 8 9` (illustrative input in nW) |
| How do I regenerate the four reference films, their numerical channels and the other simulation records? | `projections` |
| How do I render one neutral atom or all 118 in full resolution? | `projections --element Fe`, `export-atlas` |
| How can the benchmark viewer data and the earlier illustrative movies be recovered? | `observatory`, `archived-movies` |

For a complete catalogue export omit `--element`. Its CSV files distinguish 118 elements, 5,842 evaluated nuclear states and 7,139 charge-state entries including the neutrals. An empty isotope or ion wave-count field means that the archived model supplies no physical count for that state. Estimated nuclear data retain their flags. At 2.4 bar, the neon observations were used to calibrate the optical families. The other four pressure settings are held-out laboratory comparisons.

### Standard replay

Install C and C++17 compilers for the native numerical kernels. The Linux route uses GCC with OpenMP. The substrate replay also contains a Clang path for macOS. Install both `lean` and `lake` from Lean 4.30.0, and check the active version using `lean --version`. With elan, run `elan toolchain install leanprover/lean4:v4.30.0`, then prefix the replay with `ELAN_TOOLCHAIN=leanprover/lean4:v4.30.0` if your current toolchain differs. The full standard replay will call CAMB 2.0.5 from the pinned Python environment.

```sh
python replay.py all --output ../bst-standard
```

This single command runs all twelve numerical sections, rebuilds the observatory data, regenerates the four reference films and new projection inputs, exports the catalogue and recompiles the formal proofs. It writes a summary to `report.json` and individual logs beside each section. The physics replay recomputes the fourteen registered summary outputs using the supplied field and transport inputs. A checksum match tests reproducibility with the supplied inputs. The 97 cognitive tests exercise the stated models and protocols. The 84 Lean declarations have explicit premises and dependencies in `proofs/declarations.csv`.

### Long field integrations

```sh
python -m pip install -r requirements-long.txt
python replay.py long --output ../bst-long
```

This separate path rebuilds the physical fields and transport inputs used by the summary. Some three-dimensional grids reach 551 sites on a side. They require a suitably provisioned machine and substantially more memory and time than the standard replay. The long run writes its work tree and log into the selected output directory. The supplied trajectories are available to the other sections without this run.

## Use the observatory

Open `observatory/index.html` from the extracted repository. The top-level `BST_Observatory.html` in the review package provides the same two tabs as a self-contained offline file. The language selector applies to both tabs. The four reference preparations each have a French and an English film, including the writing inside the image. Changing language keeps the current time sample.

In **Simulations**, choose a domain, a preparation and a reading. Pause to compare molecular resolution or spectral filters at a fixed molecular frame. The reference view exports the original GIF with its native timings and dimensions. Every calculated view exports the same sequence drawn by the browser and the command-line renderer. Exports use a 90 ms frame delay, or 440 ms for an angular-beam step. Scientific time is labeled independently where available. The background stays black during playback.

| Simulation | Data and controls |
| --- | --- |
| Primordial wave | The reference plate follows an isolated wave and a coupled pair. The adjustable preparation supports contact stiffness, inertia, restoring stiffness and initial radial direction. White shows the local boundary speed. Amber represents energy stored at contact. |
| Substrate | Select stiffness 1, 2 or 4 and inertia 4, 8 or 16. Each combination draws one archived integration at 25 times. Shape, scalar and charge profiles share the same clock. These nine tested settings define a discrete grid. |
| Atomic atlas | Each of 118 neutrals has a full coherent film with its total, core and surface counts. A second view follows a native displacement mode. For H and the 82 heavy elements, the notes identify inferred positions. Neutral krypton has 297 sites. Its separate eight-mode preparation has 309. Isotopes and ions have exportable composition cards. The neutral film and its count remain a separate reference. No state-specific BST wave total is assigned. |
| Molecules | Kr₂ and excited C₂ have native film, internal-site and mantle readouts. C₂ has three measured band filters. H₂O, CO₂ and SF₆ share a physical scale with the same covalent radius table and neutral geometries. The model amplitudes and optical encodings are documented in the projection methods. |
| Thermal ensembles | The solid view shows reference structures for hexagonal ice, dry ice, plastic SF₆ and solid Kr with reduced harmonic motion. Two camera angles reveal their three-dimensional arrangement. Liquid, gas and ordered comparator use reduced Lennard–Jones centre trajectories. The silhouettes show motion in grey. The optical panel displays the measured quantity for the selected phase. The water and ice cells compare transmitted daylight across six path lengths. |
| Magnetic flux | Select an SF₆ mode and dipole fit to display the relative exterior magnetic induction beside the molecular motion. Six archived substrate-current runs expose time, wave number, detector direction and transverse power. Three classical fixed-dipole scenes are available for comparison. |
| Cosmology | Each of twelve CAMB families uses the same random coefficients for multipoles 48–84. All seven beams show C00, the selected family and their signed difference. Two fixed temperature scales and the residual RMS make close families comparable. |
| Gravitating bodies | Choose radius, molecular or atomic composition and number density for two or three bodies. Their aggregate masses set the Newtonian forces. Read the trajectories and energy balance through first contact or 96 hours. |

**Iodine vapour in daylight.** Open *Molecules → I₂ → Sites and mantles* and vary the optical path. Two grey iodine atlas shapes illustrate the bond vibration. The lower panel calculates what passes through a cell from the published absorption spectrum. A one-metre example is in `examples/iodine-vapour.gif`.

**Measurements and controls** retains the benchmark observatory. Its independent panel filters the 576 recorded family combinations and identifies examples from the compatible subset. The pair controls integrate a reduced dynamical system. Changing a setting in this panel does not refit the eleven-domain benchmark. Use this tab to examine the original numerical observables and sight lines.

| Measurement | Data and controls |
| --- | --- |
| Spectroscopy | Infrared absorption of SF₆ at nominal 225.5 Torr, a measured visible hydrogen doublet, or measured band-integrated powers for the gases. The hydrogen lines have equal marker heights. Their displayed hue approximates the measured wavelengths. These data do not determine a spatially resolved RGB emission field around a molecule. |
| Current and polarization | Six preparations, nine instants, four wave numbers and four directions of observation. Power is computed from transverse amplitudes. The field simulation additionally shows a relative exterior induction map inferred from a calibrated molecular dipole. |
| Thermodynamics | Coexistence observables for krypton and xenon compared with HEOS; external NIST saturation-pressure curves for water and solid carbon dioxide over their stated temperature intervals. The slider changes equilibrium temperature. |
| Cosmology | Twelve fixed CAMB families compared with Planck TT, TE and EE band powers. The cursor scans angular multipole. The comparison includes diagonal band errors and the official binned best fit. |
| Nuclear decay | Evaluated half-lives of neutral helium-6, carbon-14 and oxygen-19 determine three population survival curves. The catalogue contains 19 beta transition entries over the various charge states. The curves do not resolve the dynamics of individual events. The distribution contains no verified fission or fusion trajectory. |

The hydrogen line markers in the Measurements tab use the analytic CIE fits in [Wyman, Sloan and Shirley (2013)](https://jcgt.org/published/0002/02/01/). Thermal references for [water](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=4) and [carbon dioxide](https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=4) specify the fitting intervals. Scientific inputs and their citations are listed in Appendix D.

### Export or regenerate the films

French reference GIFs are in `examples/` and their English counterparts are in `examples/en/`. The Python regeneration uses the packaged physical inputs, the fonts and the numerical renderer. The reference films can be regenerated without Node. Install Node.js 20 or later and the pinned canvas dependency to export individual calculated GIFs from the command line. The offline page and its Export GIF button need no installation.

```sh
python replay.py projections --output ../bst-projections
npm --prefix observatory install --no-audit --no-fund
python replay.py projections --element Fe --output ../iron
python replay.py export-atlas --output ../bst-atlas
python projections/compact_atlas.py ../bst-atlas/neutral-atlas ../compact-atlas
python observatory/export_gif.py molecule C2 ../carbon-green.gif --view molecular --band 516.5
python observatory/export_gif.py molecule H2O ../water-bend.gif --view molecular --mode 1
python observatory/export_gif.py thermal H2O ../warm-ensemble.gif --view gas
python observatory/export_gif.py thermal CO2 ../dry-ice.gif --view solid --angle 38 --elevation 22 --colour spectral
python observatory/export_gif.py molecule I2 ../iodine-vapour.gif --view molecular --path-length 1
python observatory/export_gif.py thermal H2O ../water-5m.gif --view liquid --path-length 5 --colour spectral
python observatory/export_gif.py atom He ../helium-4-plus.gif --isotope 4 --nuclear-state gs --charge 1
python observatory/export_gif.py sky C01 ../beam.gif --view map
python observatory/export_gif.py field SF6 ../induction.gif --mode 11
python projections/atlas_geometry.py Og ../og-sites.csv --phase 0.25
python projections/gravity.py binary ../binary-orbit.csv
python observatory/export_gif.py substrate material ../charge-profile.gif --rigidity 4 --field-inertia 16 --quantity charge
python observatory/export_gif.py beat pair ../adjustable-pair.gif --stiffness 3 --inertia 0.3 --sign -1
```

`observatory/render.cjs` is the native batch entry point. Existing spectral, nuclear, current and quantitative spectrum commands remain available. To explicitly export a view from the measurements tab, use `python observatory/export_gif.py ... --engine comparison`. To drive the simulation tab in a browser, set `--engine browser --chromium /path/to/chromium` after installing the Python Playwright package. `python observatory/standalone.py ../BST_Observatory.html` builds the independent file. The legacy CLI value `--colour composition` opens the greyscale motion view. The projection equations, seeds, calibrated optical readouts and verification scope are documented in [Projection methods](projections/README.md).

## Inspect or extend the work

The 68 outcomes span eleven domains. The register retains 44 explicitly conditional outcomes. Of the 576 recorded choices, 192 meet the other 67 checks simultaneously within their stated scope. Chapter 7 gives three qualifications for that joint result and identifies experiments that can distinguish between compatible families. Match a new measurement to its preparation and observable before applying a tolerance.

The common-cone condition follows from the characteristic speeds of the signal and the clock model on the stated background. The strontium comparison reconstructs a clock frequency from measured atomic intervals and tests an archived contact operator. A microscopic prediction of this transition from BST remains open. In cognition, the formal and numerical results establish implications about distinguishability and retention under their premises. External agency is a philosophical extension. Converting such a claim into an empirical one requires a discriminating measurement.

The public scientific files have descriptive names, grouped by topic in `reference/sources/`. The split binary inputs belong together. `reference/stage.py` checks their hashes, restores the benchmark's internal directory structure and resolves the archived identifiers in `reference/schema-aliases.json`. Original field names, output filenames and historical reference identifiers remain available so archived tables and comparisons can be matched exactly. Extract a working copy using `python replay.py extract --output ../bst-inputs`. The archived names cited in the paper refer to `../bst-inputs/benchmark/` or to the nested ZIP identified in the register.

Begin a new scientific change in a separate working copy. Preserve the original files, record the changed preparation or law and compare with the issued checksums and criteria. In particular, changing a parameter after observing a held-out result creates a new calibration. Independent laboratories can still compare against the frozen version.

To build the two PDFs from the editable manuscripts, install Pandoc, XeLaTeX, the English and French Babel language packs, DejaVu Serif/Sans/Mono and Latin Modern Math. Then run `python paper/build.py both --output ../bst-papers`. The figure images for both languages are included. Human and AI contributions are described in Appendix D. Rights and attribution for the external data and tools follow their original sources.
