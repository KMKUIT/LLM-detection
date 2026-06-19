# LLM-detection with Bayesian Active Learning

<!-- <div align="center">
<p><img src="https://i.imgur.com/mEmBXEB.png" alt="Logo" width="300"></p>
</div> -->

**As part of:** Bachelor Thesis Artificial Intelligence;

**Built by:** Kamil Kuit.


## About the Project

This repository contains the files for the research regarding Bayesian Active Learning in LLM-detection. 

## Summary
The importance of detecting Large Language Model (LLM) generated text has become critical for the mitigation of societal issues surrounding LLMs, such as academic misconduct, fraud or information distortion. To further tackle these socially undesired byproducts of LLMs, this paper proposes improvements to an AI-generated text detection (AGTD) Bayesian surrogate model (BSM), that detects AI more efficiently than DetectGPT's state-of-the-art model. DetectGPT has shown that LLM-generated texts tend to occupy negative curvature regions of the model's log probability function. Using this information, DetectGPT detects texts with a high accuracy. The BSM adjusted and improved this method by exploring the curvature more efficiently with an uncertainty-maximizing (local) sample selection method, however they did not test the exploration acquisition function of global active learning. This research introduces this sample selection method which aims to minimize the uncertainty by estimating the sample's posterior variance (global). The results highlight increased efficiency performances of LLM-detection with global active learning strategies.


## Installation.

**To install the correct packages use:**

```
pip install -r requirements.txt
```

  

**To do a test run do:**

```
python src/run.py
```

It should return 'Likely LLM-generated: True/False'

  
