# Person_Re-Identification

Topic: `Computer Vision` 
<br>
Type: `Detection of Objects (and Re-Identification)`

## Introduction

I aim to achieve real-time person recognition in video streams, even when individuals briefly exit the camera's field of view. My objective is to accomplish this recognition with precision and without the need for individual-specific training. Moreover, I aspire to deploy this project on edge devices like the Raspberry Pi, subject to hardware feasibility.

My work focuses on addressing a practical challenge: the continuous identification of individuals in dynamic environments. Whether for enhancing security, surveillance, or innovative applications, I am developing a system that transcends the limitations of conventional recognition systems.

Within this README, you'll find comprehensive details regarding my project's goals, methodologies, dataset considerations, and my strategic work plan.

![REid Image](/Person_REid_sample.gif)

## Project Details

The primary objective of this project is to develop a robust and efficient system for person re-identification using deep learning techniques. Person re-identification, a fundamental task in computer vision and surveillance, involves recognizing and tracking individuals even when they momentarily leave the camera's field of view. I recognize that there are several deep learning methods available for this task, with some notable approaches being DeepSORT, StrongSort, and Siamese Networks. However, this project remains open to exploring and rigorously testing these methods to determine which one offers the best balance between performance and computational resources.

### Scope and Objectives

The scope of this project encompasses the following key aspects:

+ **Method Evaluation:** The initial phase of this project is dedicated to evaluating various person re-identification methods. I will implement and test approaches like DeepSORT, StrongSort, and Siamese Networks using a diverse and labeled dataset that simulates real-world scenarios. My aim is to assess the performance of each method and understand their strengths and limitations.

+ **Performance Metrics:** To objectively compare these methods, I will utilize a range of performance metrics such as identification accuracy, precision, and recall. These metrics will guide me in selecting the method that offers the highest accuracy while maintaining acceptable computational efficiency.

+ **Resource Considerations:** Beyond performance metrics, I will also assess the computational resources required by each method. Given the ever-increasing need for edge device deployment, I will evaluate the feasibility of running the selected method on hardware with limited computational capabilities, such as the Raspberry Pi.

### Methodology

+ **DeepSORT and StrongSort:** DeepSORT and StrongSort are well-established methods known for their ability to track individuals across different camera views. DeepSORT employs deep learning for feature extraction combined with the SORT (Simple Online and Realtime Tracking) algorithm for tracking. StrongSort builds upon DeepSORT's capabilities to enhance tracking and re-identification accuracy.

+ **Siamese Networks:** Siamese Networks are recognized for their effectiveness in learning similarity metrics between images. In the context of person re-identification, Siamese Networks can play a vital role in matching individuals across different camera views.

### Project Approach

+ **Method Evaluation Phase:**  I will test different Method like DeepSORT, StrongSort and Siamese Networks using a labeled dataset. Real-world scenarios will be simulated to assess the methods' performance across a variety of challenging conditions, enabling me to grasp their relative advantages and shortcomings.

+ **Performance and Resource Assessment:** Building on the evaluation phase, I will conduct a comprehensive analysis of performance metrics and computational resource requirements for each method. These insights will provide a foundation for selecting the method that strikes the optimal balance between accuracy and computational efficiency.

+ **Implementation and Deployment:** Following the method selection, I will proceed with full-scale implementation and deployment. The chosen method will undergo fine-tuning and optimization to ensure efficient operation, especially on edge devices where resource constraints may be a concern.

### Dataset 

For this project, I will be utilizing the Market-1501 Dataset. The Market-1501 Dataset is a widely recognized benchmark for person re-identification tasks. It consists of a substantial collection of images captured in a real-world scenario, featuring multiple camera views. This dataset provides a diverse and challenging set of images, making it an ideal choice for training and evaluating person re-identification models. It includes a rich variety of clothing, lighting conditions, and poses, reflecting the complexity of real-world surveillance environments. The Market-1501 Dataset will serve as the foundation for training and testing the deep learning models in this project.

link to the [Market-1501 Dataset](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)

LUPerson boasts a remarkable scale, containing over 4 million images representing in excess of 200,000 unique identities. While its size and diversity make it an enticing choice for research, it's important to note that due to its sheer magnitude, it may be less practical for this projects with resource constraints. 

link to the [LUPerson Dataset](https://github.com/DengpanFu/LUPerson)

### Goals

This project encompasses several critical objectives that span a wide range of applications:

+ **Market Research and Analysis:** One of the primary goals is to conduct comprehensive market research to gain valuable insights. This includes capturing and analyzing data, such as the count of different individuals entering a building daily. Such information can offer valuable insights to businesses and public establishments, helping them better understand customer flow and optimize marketing strategies.

+ **Surveillance and Security:** Another key objective is to enhance surveillance and security. Implementing reliable real-time person recognition and tracking can significantly enhance the efficiency of security systems. This is especially relevant in security-critical environments such as airports, train stations, and large event venues.

+ **Additional Applications:** Beyond the aforementioned objectives, the opportunities arising from this project are diverse. This includes applications in areas such as behavior analysis, traffic management, access control, and much more. The ability to identify and track individuals in various scenarios can revolutionize numerous industries and application fields.

This project is designed for versatility and innovation, with objectives continuously adjusted to meet the evolving requirements and opportunities that arise during the research and development process.

## Research

### Literature Review

To ensure a robust foundation for this project, a literature review was conducted. Two scientific papers have been meticulously examined to provide valuable insights into the topic at hand.

[Deep Learning for Person Re-identification: A Survey and Outlook:](https://arxiv.org/abs/2001.04193)
This paper offers an extensive exploration of the fundamental facets of person re-identification (Re-ID) and serves as a valuable reference for fostering a profound comprehension of this subject. The paper in focus delves into the dynamic realm of person re-identification (Re-ID), which is paramount within the realms of multi-camera surveillance and deep learning. The authors classify Re-ID into closed-world and open-world scenarios, placing a primary emphasis on the latter due to its practical relevance. This work conducts a thorough examination of the components and techniques employed in Re-ID, encompassing deep feature representation, metric learning, and ranking optimization. Additionally, it introduces a potent AGW baseline that demonstrates state-of-the-art or competitive results across twelve datasets. The paper also introduces a novel evaluation metric (mINP) for assessing the practical applicability of Re-ID systems. Furthermore, it elucidates significant unresolved challenges in the field, paving the way for further exploration in this ever-evolving domain.

[Unsupervised Pre-training for Person Re-identification:](https://arxiv.org/abs/2012.03753v2)
This paper introduces "LUPerson" a substantial unlabeled person re-identification dataset. It pioneers unsupervised pre-training on this dataset, achieving state-of-the-art performance on key Re-ID datasets, with particular gains on smaller-scale and few-shot scenarios. Notably, this paper currently holds the top performance on the Market1501 dataset.

## Work Breakdown structure

### Dataset Collection (1 days)
+ Research and select the dataset (0.5 day)
+ Quality control and dataset cleaning (0.5 day)

### Designing and Building the Network (5 days)
+ Research network architectures suitable for Re-ID (1 day)
+ Network design and architecture planning (1 days)
+ Implementation of the chosen network architecture (2 days)
+ Integration of pre-trained models, if applicable (1 day)

### Training and Fine-Tuning the Network (4 days)
+ Training the network (2 days)
+ Hyperparameter tuning and optimization (2 days)
+ Model fine-tuning based on validation results (1 day)

### Building an Application to Present Results (1 days)
+ Design and development of the application (1 days)
+ Integration with the trained model (1 day)

### Writing the Final Report (4 days)
+ Preparing the report structure and outline (0.5 day)
+ Literature review and research methodology (1 day)
+ Results and analysis (1.5 days)
+ Conclusion and discussion (1 day)

### Preparing the Presentation of Your Work (2 days)
+ Preparing the presentation outline and structure (0.5 day)
+ Creating presentation slides (1 days)
+ Rehearsing and refining the presentation (0.5 day)
