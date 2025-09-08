

# Tiny Moves: Game-based Hypothesis Refinement

**Authors:** Rogier Hintzen*, Agnieszka Dobrowolska*, Martin Balla*, Karl Gemayel, Sabine Reichert, Thomas Charman, Anna Gogleva  
*These authors contributed equally.*

**Workshop:** AI for Science Workshop @ NeurIPS 2025

## Abstract

Scientific discovery is an iterative process, yet most machine learning approaches treat it as an end-to-end prediction task, limiting interpretability and alignment with scientific reasoning workflows. We introduce The Hypothesis Game, a symbolic, game-based framework where a system of agents refines hypotheses through a fixed set of reasoning moves (a reasoning grammar). Inspired by the idea that scientific progress often relies on small, incremental changes, our framework emphasizes “tiny moves” as the building blocks of incremental hypothesis evolution. We evaluate the approach on pathway-level reasoning tasks derived from Reactome, focusing on reconstruction from partial cues and recovery of corrupted hypotheses. Across 820 reconstruction and 2880 corruption experiments, it matches strong prompting baselines on reconstruction and achieves superior precision and error recovery in corruption. Beyond accuracy, it produces concise, interpretable hypotheses and enables controllable reasoning, highlighting the potential of game-based reasoning for scientific discovery.

## Repository Structure

This repository contains the code and resources for the NeurIPS paper:

- `tiny_moves/` - Main package source code
- `requirements.txt` - Python dependencies
- `README.md` - This file

Key submodules:
- `agents/` - Agent implementations
- `entry_points/` - Entry scripts
- `metrics/` - Evaluation metrics
- `operations/` - Operation definitions and YAML configs
- `representations/` - Data representations
- `retrieval/` - Retrieval utilities
- `state/` - State management
- `tools/` - Tooling and registry
- `trajectory/` - Trajectory logic
- `utils/` - Utility functions

## Installation

Clone the repository and install dependencies:

```bash
git clone [REPO_URL]
cd tiny_moves_private
pip install -r requirements.txt
```

## Usage

Example usage:

```bash
# [Add example command to run your main script or experiment]
export OPENAI_API_KEY="your_openai_api_key"
python  tiny_moves/entry_points/chat.py --config_name tiny_moves_no_corpus.yaml 
```


## Citation

If you use this code for your research, please cite our paper:

```bibtex
@inproceedings{hintzen2025tinymoves,
	title={Tiny Moves: Game-based Hypothesis Refinement},
	author={Hintzen, Rogier and Dobrowolska, Agnieszka and Balla, Martin and Gemayel, Karl and Reichert, Sabine and Charman, Thomas and Gogleva, Anna},
	booktitle={Proceedings of the Neural Information Processing Systems (NeurIPS), AI for Science Workshop},
	year={2025},
	url={https://github.com/RelationRx/tiny_moves_private}
}
```

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

## Contact

For questions or collaborations, please contact [corresponding author email].
