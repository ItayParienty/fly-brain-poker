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
