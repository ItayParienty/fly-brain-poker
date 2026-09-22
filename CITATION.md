# Data attribution

This project does not include any connectome data. It downloads the public
FlyWire FAFB v783 snapshot at runtime via `download_data.py`, from
`https://storage.googleapis.com/flywire-data/codex/data/fafb/783/`.

If you use this code or the data it fetches, cite the original work:

> Dorkenwald, S., Matsliah, A., Sterling, A. R., et al. (2024).
> *Neuronal wiring diagram of an adult brain.* Nature 634, 124-138.

> Schlegel, P., Yin, Y., Bates, A. S., et al. (2024).
> *Whole-brain annotation and multi-connectome cell typing of Drosophila.*
> Nature 634, 139-152.

FlyWire data is released under CC BY-NC-SA 4.0 for non-commercial use.
See https://flywire.ai/guidelines for the current terms.

Neurotransmitter predictions in the dataset are from:

> Eckstein, N., Bates, A. S., Champion, A., et al. (2024).
> *Neurotransmitter classification from electron microscopy images at
> synaptic sites in Drosophila melanogaster.* Cell 187, 2574-2594.

## Biology the simulation is based on

- Turner, G. C., Bazhenov, M., Laurent, G. (2008). *Olfactory representations
  by Drosophila mushroom body neurons.* J Neurophysiol 99, 734-746.
  — the 5-10% Kenyon cell sparseness this project calibrates against.
- Lin, A. C., Bygrave, A. M., de Calignon, A., et al. (2014). *Sparse,
  decorrelated odor coding in the mushroom body enhances learned odor
  discrimination.* Nat Neurosci 17, 559-568.
  — the role of APL inhibition in keeping that code sparse.
- Hige, T., Aso, Y., Modi, M. N., et al. (2015). *Heterosynaptic plasticity
  underlies aversive olfactory learning in Drosophila.* Neuron 88, 985-998.
  — dopamine-gated depression of Kenyon cell to MBON synapses.
- Aso, Y., Sitaraman, D., Ichinose, T., et al. (2014). *Mushroom body output
  neurons encode valence and guide memory-based action selection.* eLife 3,
  e04580.
  — MBONs as opposing approach/avoid drives.

## The eye

The optic lobe's cell types and columns come from FlyWire's visual system
atlas (Matsliah et al. 2024, *Neuronal parts list and wiring diagram for a
visual system*, Nature 634, 166-180). Time constants, resting potentials and
per-type synaptic strengths are taken from the published flyvis models:

> Lappalainen, J. K., Tschopp, F. D., Prakhya, S., et al. (2024).
> *Connectome-constrained networks predict neural activity across the fly
> visual system.* Nature 634, 1132-1140.

## The game

Bloons Tower Defense is a game by Ninja Kiwi (2007). `bloons/` is an
independent, non-commercial reimplementation for research; it is not
affiliated with or endorsed by Ninja Kiwi. Its rules and geometry were read
from the original game file, which is not included here.
