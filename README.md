# Person_Re-Identification

Topic: `Computer Vision` 
<br>
Type: `Detection of Objects (and Re-Identification)`

## Introduction

I aim to achieve real-time person recognition in video streams, even when individuals briefly exit the camera's field of view. My objective is to accomplish this recognition with precision and without the need for individual-specific training. Moreover, I aspire to deploy this project on edge devices like the Raspberry Pi, subject to hardware feasibility.

My work focuses on addressing a practical challenge: the continuous identification of individuals in dynamic environments. Whether for enhancing security, surveillance, or innovative applications, I am developing a system that transcends the limitations of conventional recognition systems.

Within this README, you'll find comprehensive details regarding my project's goals, methodologies, dataset considerations, and my strategic work plan.

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

+ **Method Evaluation Phase:** In the project's initial phase, I will meticulously implement and rigorously test DeepSORT, StrongSort, and Siamese Networks using a labeled dataset. Real-world scenarios will be simulated to assess the methods' performance across a variety of challenging conditions, enabling me to grasp their relative advantages and shortcomings.

+ **Performance and Resource Assessment:** Building on the evaluation phase, I will conduct a comprehensive analysis of performance metrics and computational resource requirements for each method. These insights will provide a foundation for selecting the method that strikes the optimal balance between accuracy and computational efficiency.

+ **Implementation and Deployment:** Following the method selection, I will proceed with full-scale implementation and deployment. The chosen method will undergo fine-tuning and optimization to ensure efficient operation, especially on edge devices where resource constraints may be a concern.

### Dataset 

For this project, I will be utilizing the Market-1501 Dataset. The Market-1501 Dataset is a widely recognized benchmark for person re-identification tasks. It consists of a substantial collection of images captured in a real-world scenario, featuring multiple camera views. This dataset provides a diverse and challenging set of images, making it an ideal choice for training and evaluating person re-identification models. It includes a rich variety of clothing, lighting conditions, and poses, reflecting the complexity of real-world surveillance environments. The Market-1501 Dataset will serve as the foundation for training and testing the deep learning models in this project.

link to the [Market-1501 Dataset](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
