
# Human Identification and Verification using Gait Analysis and Facial Recognition

## Executive Summary
The COVID-19 pandemic has accelerated the need for contactless authentication technologies. Traditional methods like fingerprint scanning and facial recognition face challenges due to the need for physical contact and high setup costs. This project proposes a novel access control system that combines gait analysis and facial recognition, providing a secure, cost-effective, and contactless solution. By leveraging Kinect's depth-capturing capabilities and a regular camera for facial recognition, our system achieves a combined accuracy of 89%, making it a robust alternative to more expensive, complex systems.

## Business Problem
In today's security-conscious environment, there is a growing demand for contactless authentication systems, particularly in sensitive areas such as healthcare, airports, and corporate offices. Traditional biometric systems like fingerprint scanners and facial recognition have limitations, including the need for physical contact and susceptibility to environmental factors. The business problem addressed by this project is the need for a reliable, secure, and cost-effective contactless authentication system that can operate effectively in various settings.

## Methodology
The project is divided into three phases:
1. **Recording Phase**: Using a Kinect depth camera, we captured gait cycles of individuals. Facial images were recorded using a regular webcam.
2. **Preprocessing**: The recorded images were processed to create Gait Energy Images (GEI) and facial embeddings. This involved using advanced image processing techniques to develop unique biometric signatures for each individual.
3. **Live Implementation**: The system performs real-time gait and facial recognition by comparing live data with stored embeddings in a database. Access is granted if a match is found; otherwise, the system denies entry.

## Skills
- **Computer Vision**: Utilized for processing and analyzing images to create Gait Energy Images and facial embeddings.
- **Machine Learning**: Applied Convolutional Neural Networks (CNN) to analyze gait and facial features for identification and verification.
- **Image Processing**: Techniques such as color inversion and superimposition were used to extract unique gait and facial features.
- **Python Programming**: Implemented the system using Python libraries such as OpenCV, Keras, and Pillow.

## Results
The project successfully demonstrated that combining gait analysis with facial recognition significantly improves the accuracy of contactless authentication systems. The system achieved an overall accuracy of 89%, with a True Positive Rate (TPR) of 98% for gait recognition and 80% for facial recognition. The implementation also proved to be cost-effective compared to existing high-end systems.

## Next Steps
The project has laid the groundwork for a reliable contactless authentication system. Future work could involve:
- **Improving Gait Recognition**: Further refining the gait analysis algorithms to handle variations in clothing and footwear.
- **Scalability**: Expanding the system to handle larger datasets and more complex environments.
- **Integration**: Exploring integration with other biometric systems to enhance overall security.
- **Commercialization**: Investigating potential market applications in industries such as healthcare, security, and retail.
