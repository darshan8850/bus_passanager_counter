# Person Head Detection for Passenger Counter in Bus

https://github.com/user-attachments/assets/d53f7175-93f3-431c-88d7-e2e744d81513





## Overview

This project aims to develop an efficient and accurate system for counting passengers in a bus by detecting their heads using computer vision techniques. The system leverages the **SSD MobileNetV2** model to identify and track passengers as they board and alight, ensuring precise passenger counts in real-time.

## Features

- **Head Detection with SSD MobileNetV2**: Utilizes the SSD MobileNetV2 model, known for its balance between speed and accuracy, to detect and count the heads of passengers in real-time, even in crowded environments.
- **Real-Time Processing**: The system is optimized for real-time operation, ensuring up-to-date passenger counts as the bus operates.
- **Robust to Variability**: Handles various lighting conditions, different head orientations, and varying passenger heights.
- **Scalability**: Designed to be easily integrated into existing bus surveillance systems with minimal hardware requirements.
- **Data Logging**: Logs passenger count data for further analysis and reporting.

## SSD MobileNetV2

- **Architecture**: Single Shot Multibox Detector (SSD) with MobileNetV2 backbone.
- **Pre-trained Weights**: The model is pre-trained on the COCO dataset and fine-tuned for head detection.
- **Advantages**: SSD MobileNetV2 offers a great trade-off between speed and accuracy, making it ideal for deployment in environments like buses where real-time processing is crucial.
