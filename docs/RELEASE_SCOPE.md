# Scope and file map

`rrq/quantization.py`: explicit final representation and integer oracle.
`rrq/linear.py`: seven-projection model conversion and native dispatch.
`rrq/calibration.py`: original 128x20 bounded search equations.
`rrq/csrc/`, `kernels/`: final native sources, no cached shared objects.
`rrq/runtime/`: dependencies of the selected common runtime. Internal
layers of the graph implementation are necessary modules, not abandoned methods.
`artifacts/scales/`: nine exact quantizer-matched final scale files, three bits
by three dense 8B models; no weight checkpoints or executable pickle artifacts.
`data/`: public example identifiers, text, preprocessing metadata and hashes.
`configs/`: final method and evaluation settings.
`results/`: final numeric tables, population diagnostics and layer-wise values.
`scripts/`: portable inference, capture/search, manifest evaluation, timing and gates.

The package consolidates the final method instead of copying the historical
working tree. Porting launchers were replaced with relative-path interfaces.
It excludes experimental queues, server state, pinned old filesystem receipts,
abandoned candidates, author-facing notes, SSH information, source PDFs, generated
figures, full response archives, model/feature dumps, and binary build products.

Release verification checks mathematical parity of selected operators and
artifact preservation. It does not assert that a full new 8-task run, complete
fresh calibration, or every external comparison method was executed again from
this cleaned package. Those expensive experiments are preserved as exported
final observations, with the supplied scales and documented public data sources.
