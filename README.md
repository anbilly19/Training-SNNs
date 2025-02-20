
### Project Title: Analysis of Direct Training of Spiking Neural Networks

#### Objective
To implement and evaluate the performance of Spiking Neural Networks (SNNs) in comparison to traditional neural networks, focusing on efficiency and computational power in processing temporal data.

#### Methodology
- **Data Preprocessing**: Collected and preprocessed datasets suitable for training SNNs.
- **Model Selection**: Chose appropriate SNN architectures and parameters.
- **Optimization Techniques**: Applied various optimization algorithms to enhance the training process.
- **Performance Evaluation**: Conducted experiments to compare SNNs with traditional neural networks in terms of accuracy, speed, and computational efficiency.

# Vision Transformer-FMNIST
Scratch Pytorch implementation of Vision Transformer (ViT).

The network is a scaled-down version of the original architecture from [An Image is Worth 16X16 Words](https://arxiv.org/pdf/2010.11929.pdf)

<br>
Transformer Config:

 | <!-- -->    | <!-- -->    |
--- | --- | 
Input Size | 28 |
Patch Size | 4 | 
Sequence Length | 7*7 = 49 |
Embedding Size | 96 | 
Num of Layers | 6 | 
Num of Heads | 4 | 
Forward Multiplier | 2 | 

# Spiking Vision Transformer

Entry point -> run.sh <br>
3 modes: Train, Single batch test(needs to load model), Full test (all batches).<br>
All options and args are located in main.py<br>
Tensorboard events stored under logs.

#### Key Findings
- SNNs demonstrated superior efficiency and computational power in tasks requiring temporal data processing.
- Traditional neural networks outperformed SNNs in certain areas, indicating the need for further research and improvements in SNN robustness and scalability.

#### Challenges
- Training SNNs effectively posed significant challenges, requiring innovative approaches and techniques.
- Ensuring the scalability of SNNs for broader applications remains an ongoing research area.

#### Conclusion
The project revealed the potential of SNNs in specific applications, particularly where efficiency and temporal data processing are crucial. However, additional research is needed to address the challenges and enhance the overall performance and scalability of SNNs.

#### Future Work
- Investigate advanced optimization techniques to improve the training of SNNs.
- Explore new architectures and models to enhance the robustness and scalability of SNNs.
- Conduct further experiments to identify and overcome the limitations of SNNs in various applications.
