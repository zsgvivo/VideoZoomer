# VIDEOZOOMER: REINFORCEMENT-LEARNED TEMPORAL FOCUSING FOR LONG VIDEO REASONING
<a href='https://www.arxiv.org/abs/2512.22315'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/zsgvivo/videozoomer'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>

This repository releases the official code for VideoZoomer.  The training code is based on [verl](https://github.com/volcengine/verl).

## Install
Please follow the instructions below to install the required packages.
```sh
# Clone this repository
git clone https://github.com/zsgvivo/VideoZoomer
# Install Package
conda create -n VideoZoomer python=3.11 -y
conda activate VideoZoomer
cd VideoZoomer
pip3 install -r requirements.txt
pip3 install -e .
pip3 install httpx==0.23.3
```

## Training

### SFT
We use LLaMA-Factory to perform SFT on our cold-start data. We use Qwen2.5-VL-7B-Instruct as the base model.

Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

Download the [SFT dataset](https://huggingface.co/datasets/zsgvivo/videozoomer_sft) from huggingface.

Follow the instructions in LLaMA-Factory to configure the cold-start data in data/dataset_info.json, 
run the following command to train your model.
```sh
llamafactory-cli train scripts/sft.yaml
```


### RL
The reinforcement learning is based on the cold-start model. Download the [RL dataset](https://huggingface.co/datasets/zsgvivo/rl_datasets) from huggingface and run `bash scripts/train_video.sh` to train the model.

## Evaluation
The evaluation is also based on verl framework, set `trainer.val_only=True` in the script to conduct evaluation. Run `bash scripts/eval_video.sh` to evaluate the model with tool call, and `scripts/eval_video_notool.sh` to evaluate the model without tool call.

Change `actor_rollout_ref.rollout.max_generation_round`, `actor_rollout_ref.rollout.tool_call_max_frames` to evaluate the model with different max generation round and toolmax frames per turn.


## Citation
If you find this repo useful for your research, please consider citing the paper
```
@article{videozoomer,
    title={VideoZoomer: Reinforcement-Learned Temporal Focusing for Long Video Reasoning}, 
    author={Yang Ding and Yizhen Zhang and Xin Lai and Ruihang Chu and Yujiu Yang},
    year={2025},
    journal={arXiv:2512.22315},
    url={https://arxiv.org/abs/2512.22315}, 
}
```
