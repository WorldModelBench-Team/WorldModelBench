# WorldModelBench

[**🌐 Homepage**](https://worldmodelbench-team.github.io/) | [**🏆 Leaderboard**](https://worldmodelbench-team.github.io/#leaderboard) | [**📖 WorldModelBench arXiv**](https://arxiv.org/pdf/2502.20694)

This repo contains the evaluation instructions for the paper "[WorldModelBench: Judging Video Generation Models As World Models](https://arxiv.org/pdf/2502.20694)".

## 🔔News
- **🔥[2025-07-28]: We have moved the data from huggingface to github for easier documentation.**
- **🔥[2025-05-21]: WorldModelBench has been accepted as an oral paper in the CVPR 2025 WorldModelBench workshop. 😆**
- **🔥[2025-03-02]: Our [WorldModelBench](https://worldmodelbench-team.github.io/) is now available. We look forward to your participation! 😆**

## Introduction

### WorldModelBench

WorldModelBencha is a benchmark designed to evaluate the **world modeling capabilities** of video generation models across **7** application-driven domains (spanning from Robotics, Driving, Industry, Human Activities, Gaming, Animation, and Natural) and **56** subdomains. Each domain features 50 carefully curated prompts, comprising a text description and an initial video frame, tailored for video generation. We provide a **human-aligned** VLM (Vision-Language Model) based judger to automatically evaluate model-generated videos on **Instruction Following**, **Common Sense**, and **Physical Adherence**.

![Alt text](worldmodelbench.png)

## Evaluation

🎯 Please refer to the following instructions to evaluate with WorldModelBench:
- **Environment Setup**: Clone and install VILA by following the instructions in [VILA Installation Guide](https://github.com/NVlabs/VILA?tab=readme-ov-file#installation).
- **Data&Model Preparation**: Download the [judge](https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b).
```
└── worldmodelbench
    └── images (first frames of videos)
    └── evaluation.py (evaluation script)
    └── worldmodelbench.json (test set)
    ...
```
The ```worldmodelbench.json``` has a list of dict containing instances for video generation.
```
[
  {
          "domain": "autonomous vehicle",
          "subdomain": "Stopping",
          "text_first_frame": "The autonomous vehicle approaches a traffic light on a bridge surrounded by tall buildings. Construction barriers line the sides of the bridge with a yellow traffic light visible ahead.",
          "text_instruction": "The autonomous vehicle stops at the traffic light on the bridge.",
          "first_frame": "images/69620089860948e38a4921dd4869d24f.jpg"
      }
...
]
```
- **Video Generation**: Perform video generation model inference. You will find 350 test instances in the worldmodelbench.json file. For *each instance*, follow the instructions below to generate the videos:
  - **Text-to-Video**: Use the following as the generation prompt:
    - Text: ```" ".join([instance["text_first_frame"], instance["text_instruction"]])```.
  - **Image-to-Video**: Use the following as the generation prompt:
    - Image: ```instance["first_frame"]```, Text: ```instance["text_instruction"]```.

  **Note**: Please save the video using the **same name** as ```instance["first_frame"]```, replacing the file extension ```.jpg``` with ```.mp4```.
- **Evaluation**: Run the following script in the ```worldmodelbench``` folder, specifying the video generation model using ```--model_name```. Upon completion, a ```worldmodelbench_results.json``` file will be generated in the same folder.
```
python evaluate.py --model_name TESTED_MODEL --video_dir GENERATED_VIDEOS --judge PATH_TO_JUDGE
```

The answers and explanations for the test set questions are withheld. You can submit your results to worldmodelbench.team@gmail.com to be considered for the leaderboard.

## Disclaimers
The guidelines for the annotators emphasized strict compliance with copyright and licensing rules from the initial data source, specifically avoiding materials from websites that forbid copying and redistribution. 
Should you encounter any data samples potentially breaching the copyright or licensing regulations of any site, we encourage you to [contact](#contact) us. Upon verification, such samples will be promptly removed.

## Contact
- Dacheng Li: dacheng177@berkeley.edu
- Yunhao Fang: yuf026@ucsd.edu
- Song Han: songhan@mit.edu
- Yao Lu: jasonlu@nvidia.com

## Citation

**BibTeX:**
```bibtex
@article{Li2025WorldModelBench,
  title={WorldModelBench: Judging Video Generation Models As World Models},
  author={Dacheng Li and Yunhao Fang and Yukang Chen and Shuo Yang and Shiyi Cao and Justin Wong and Michael Luo and Xiaolong Wang and Hongxu Yin and Joseph E. Gonzalez and Ion Stoica and Song Han and Yao Lu},
  year={2025},
}
```
This website is adapted from [MMMU](https://github.com/MMMU-Benchmark/MMMU).
