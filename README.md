# CS726_OOD


Ami Pandat – 23v0006 Comprehensive Project Proposal Title: "Out-of-Distribution Detection for Drone and Bird Differentiation in Aerial Surveillance" 


The rapid proliferation of drones poses new challenges and opportunities for airspace monitoring, wildlife study, and security protocols [3]. Deep neural networks have shown exceptional performance in various domains, including aerial object recognition. A critical challenge in this field is the confident misclassification of unseen aerial objects or classes, which the model has not encountered during training. An ideal model would withhold answers when faced with such unknown aerial classes, thereby avoiding errors in critical applications like air traffic control, surveillance, and unmanned aerial vehicle navigation. There has been progress towards training classifiers with Out-of-Distribution (OOD) samples near the in-distribution boundary. However, these efforts do not entirely encapsulate the in-distribution boundary, leading to less than optimal OOD detection. Generative Adversarial Networks (GANs) are the forefront of anomaly detection in high-dimensional data. Nevertheless, the traditional GAN loss function is not wholly suitable for anomaly detection since it aims for the generated samples' distribution to mimic real data, which can make the discriminator less effective in recognizing anomalies. Traditional OOD approaches leverage scoring functions from softmax outputs of discriminative models, which have been shown to assign misleadingly high confidence to far-removed inputs [1]. Building on the foundation laid by [2], this project proposes an energy-based scoring system for OOD detection, which aligns theoretically with input probability densities and offers the flexibility of fine-tuning or direct application on pre-trained models. This approach is particularly suited to addressing the complex dynamics and varied appearances in aerial surveillance, promising a significant advancement in the accurate classification of drones (OOD) versus birds (ID).
Technical Challenges
1. Balancing ID and OOD Performance: Ensuring the model accurately distinguishes OOD data (drones) without compromising ID data (birds) performance.
2. Dataset Diversity and Availability: The lack of large, diverse datasets for OOD scenarios, especially for drones, hinders training and evaluation efforts.
3. Adversarial Robustness: The model must resist misclassifying adversarially altered ID images as OOD, a common pitfall in security-sensitive applications. Technical Focus and Innovations
1. Auxiliary OOD Data Utilization: Addressing the gap in performance between no/limited auxiliary data settings and those benefiting from extensive finetuning on
datasets like TinyImages. The project will explore alternative sources of OOD data and novel data augmentation techniques to enhance model robustness.
2. Loss Function Design: Innovating beyond the squared hinge-loss proposed in [2] by developing loss functions that minimize sensitivity to hyperparameters and enforce a larger energy gap between ID and OOD samples. This includes investigating loss functions that can dynamically adjust based on the nature of the data and the distribution of energy scores.
3. Semantic and Structural Data Analysis: Examining the effectiveness of the energy-based OOD detection in scenarios where ID and OOD distributions are closely related, utilizing datasets with inherent semantic relationships to test model precision.
4. Enhancing Adversarial and Degradation Robustness: The project will explore mechanisms to ensure model resilience against adversarial attacks and common data degradations, such as noise and variations in image quality, which are pivotal in real-world aerial surveillance.
5. Distance-Based Energy Scoring: Developing energy functions based on distance metrics to improve the detection of OOD instances, leveraging the spatial relationships and feature distributions unique to aerial images.
6. Complexity and Type of Data: Analyzing how the complexity of in-distribution data influences OOD detection performance, particularly examining the variance in model effectiveness across datasets of differing resolutions and content complexity.

8. Datasets for Exploration
• In-Distribution (Drones): Sourcing from UAV123, VisDrone, and potential custom-generated datasets to represent a wide range of OOD scenarios
• Out-of-Distribution (Birds): Utilization of established bird datasets such as NABirds, CUB-200-2011, and the iNaturalist dataset
• Auxiliary Datasets: Employing additional datasets like SVHN and LSUN for refining the model's OOD detection capabilities in preparatory stages.

References
• [1] Hein, M., Andriushchenko, M., & Bitterwolf, J. (2019). Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. CVPR 2019.
• [2] Liu, W., Liang, X., & Mahadevan, V. (2020). Energy-based Out-of-distribution Detection. NeurIPS 2020.
• [3] Marco A. Contreras-Cruz, Fernando E. Correa-Tome, Rigoberto Lopez-Padilla, Juan-Pablo Ramirez-Paredes, Generative Adversarial Networks for anomaly detection in aerial images, Computers and Electrical Engineering, Volume 106, 2023, 108470,ISSN 0045-7906, https://doi.org/10.1016/j.compeleceng.2022.108470.
