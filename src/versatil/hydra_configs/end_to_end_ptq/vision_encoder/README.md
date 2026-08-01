# Vision encoder compression presets

This folder contains starter post-training compression configs for vision encoder-only workflows.

Replace `encoding_pipeline.encoders.<encoder_name>` with the actual encoder submodule path in your loaded policy.

Files:
- `bn_fusion_only.yaml`: batch-norm preparation only
- `prune_only.yaml`: structured + unstructured pruning only
- `ptq_only.yaml`: PT2E quantization only
- `bn_fusion_prune.yaml`: BN fusion plus structured/unstructured pruning
- `bn_fusion_prune_ptq.yaml`: BN fusion, pruning, and PT2E quantization for the targeted encoder

If you have multiple vision encoders, create one config per encoder or duplicate the module target list with distinct `module_path` values.
