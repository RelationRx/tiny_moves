

# Tiny Moves: Game-based Hypothesis Refinement

**Authors:** Agnieszka Dobrowolska*, Rogier Hintzen*, Martin Balla*, Karl Gemayel, Sabine Reichert, Thomas Charman, Jen Ning Lim, Lindsay Edwards, Anna Gogleva
*These authors contributed equally.*


## Abstract

Most machine learning approaches to scientific discovery frame hypotheses as end-to-end predictions, obscuring the incremental structure of scientific reasoning. We propose The Hypothesis Game, a symbolic formalism for hypothesis refinement in which LLM agents operate on a shared hypothesis state using a fixed grammar of reasoning moves. The framework is motivated by the observation that scientific progress often proceeds through small, localized revisions, grounded in domain context, rather than extensive rewrites. We instantiate a minimal game with LLM agents and evaluate it on pathway-level mechanistic refinement tasks. In the primary setting of corruption recovery, where hypotheses contain controlled errors, the game-based approach consistently removes more errors and achieves higher precision than strong prompting baselines, while preserving valid structure through incremental edits. In a secondary reconstruction setting from partial cues, it performs comparably to the strongest baseline, indicating that explicit move-based refinement remains competitive even when ground-truth recovery is difficult. These findings support game-based reasoning as a principled route to more controllable, interpretable, and transferable hypothesis refinement systems for scientific discovery.

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
@misc{dobrowolska2026tinymovesgamebasedhypothesis,
      title={Tiny Moves: Game-based Hypothesis Refinement}, 
      author={Agnieszka Dobrowolska and Rogier Hintzen and Martin Balla and Karl Gemayel and Sabine Reichert and Thomas Charman and Jen Ning Lim and Lindsay Edwards and Anna Gogleva},
      year={2026},
      eprint={2602.09801},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2602.09801}, 
}
```

## License

Licensed under [MIT License](LICENSE). 

## Contact

For questions or collaborations, please contact anna.gogleva@relationrx.com.
