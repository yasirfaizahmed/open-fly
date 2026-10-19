# Citing data and models used by openfly

openfly's **code** is MIT-licensed. The **connectome data** it downloads is
not part of openfly; it remains under its publishers' terms and should be
cited when you use it. `brain.meta["citation"]` and `brain.meta["license"]`
repeat this for each loaded brain.

## FlyWire (`flywire_783`, `flywire_630`)

* Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634, 124-138.
* Schlegel, P. et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634, 139-152.
* Matsliah, A. et al. (2024). Neuronal parts list and wiring diagram for a visual system. *Nature* 634, 166-180.
* Eckstein, N. et al. (2024). Neurotransmitter classification from electron microscopy images at synaptic sites in *Drosophila melanogaster*. *Cell* 187, 2574-2594.
* Buhmann, J. et al. (2021). Automatic detection of synaptic partners in a whole-brain *Drosophila* electron microscopy data set. *Nature Methods* 18, 771-774.
* Zheng, Z. et al. (2018). A complete electron microscopy volume of the brain of adult *Drosophila melanogaster*. *Cell* 174, 730-743.
* Annotation updates: Berg et al. (2026) *Cell* (male CNS cross-matching) and Tastekin et al. (2026) *Cell* (gustatory neurons), via github.com/flyconnectome/flywire_annotations.
* Terms of use: see https://codex.flywire.ai.

## The whole-brain LIF model and connectivity table

* Shiu, P. K. et al. (2024). A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634, 210-219.
  Code and data: github.com/philshiu/Drosophila_brain_model (MIT).

## Other connectomes

* BANC: Bates, A. S. et al. (2025). Distributed control circuits across a brain-and-cord connectome. *bioRxiv*.
* MANC: Takemura, S. et al. (2024) *eLife*; Marin, E. C. et al. (2024) *eLife*; Cheong, H. S. J. et al. (2024) *eLife*.
* Male CNS: Berg, S. et al. (2026). Sexual dimorphism in the complete *Drosophila* male central nervous system connectome. *Cell*.
* Male optic lobe: Nern, A. et al. (2025). Connectome-driven neural inventory of a complete visual system. *Nature*.
* Hemibrain: Scheffer, L. K. et al. (2020). A connectome and analysis of the adult *Drosophila* central brain. *eLife* 9, e57443.

## Model parameters

Kakaria & de Bivort 2017 (*Front. Behav. Neurosci.*); Jürgensen et al. 2021
(*Neuromorph. Comput. Eng.*); Lazar et al. 2021 (*eLife*); Paul et al. 2015
(*Front. Cell. Neurosci.*). See `openfly/sim/params.py`.

## Citing openfly

If openfly helped your work, please cite the data and model papers above,
and mention openfly (https://github.com/yasirfaizahmed/open-fly).
