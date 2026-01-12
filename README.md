# 🌱 Smart Nature

**An Intelligent Greenhouse Prototype for Automated Plant Care and Interactive Monitoring**

## Overview

**Smart Nature** is an intelligent greenhouse prototype designed to automate plant irrigation and enable interactive plant monitoring using sensor-driven control and AI-assisted analysis. The project addresses a common and critical problem in greenhouse and home gardening environments: **manual watering based on guesswork**, which is time-consuming, error-prone, and can lead to plant stress or death.

By integrating **soil humidity sensors**, **automated water pumping**, **camera-based monitoring**, and an **AI-powered RAG (Retrieval-Augmented Generation) system**, Smart Nature creates a semi-autonomous ecosystem where plants are watered only when necessary and users can interact with plant health data intelligently.

---

## Problem Statement

In conventional greenhouses:

* Plants require daily watering based on soil conditions.
* Manual monitoring is inconvenient and inconsistent.
* Overwatering or underwatering increases the risk of plant damage.
* Continuous visual inspection is impractical.

These challenges motivated the development of **Smart Nature**, which automates irrigation and introduces intelligent monitoring with minimal human intervention.

---

## Key Features

* **Automated Irrigation System**

  * Soil humidity sensors continuously monitor moisture levels.
  * A water pump is automatically activated when soil humidity drops below a defined threshold.
  * Ensures optimal watering and reduces human error.

* **Smart Monitoring with Camera & Servo Motor**

  * A camera mounted on a servo motor provides rotational plant monitoring.
  * Captures visual data of plant conditions from multiple angles.

* **AI-Driven Plant Interaction (RAG System)**

  * Camera data is sent to a Retrieval-Augmented Generation system.
  * Two external APIs are used to analyze plant condition and generate responses.
  * Users can interact with the system and receive insights based on observed plant health.

* **Prototype-Oriented Design**

  * Hardware-integrated solution using Arduino.
  * Modular architecture allows future expansion (e.g., real-time object detection).

---

## System Architecture (High-Level)

1. **Soil Humidity Sensors** collect moisture data.
2. **Arduino Controller** processes sensor signals.
3. **Water Pump Motor** activates automatically when soil moisture is low.
4. **Camera + Servo Motor** captures plant visuals.
5. **Database & RAG System** receive and analyze visual data.
6. **User Interface (Indirect)** enables intelligent interaction with plant conditions.

---

## Methodology

### Dataset Collection

The dataset was created by collecting plant images from:

* Home gardens
* University campus plants
* Local gardens
* Internet sources

This ensured diversity in lighting, plant species, and health conditions.

### Model Training & Comparison

Multiple models were trained and evaluated to analyze plant conditions:

* **Elastic Net**
* **SqueezeNet**
* **Small Custom CNN**

### Model Performance

After comparative evaluation:

* **Elastic Net** demonstrated the best overall performance for the given dataset and constraints.
* It offered a strong balance between accuracy, generalization, and computational efficiency, making it suitable for prototype deployment.

---

## Hardware Components

* Arduino microcontroller
* 2 × Soil humidity sensors
* Water pump motor
* Water tank
* Servo motor (for camera rotation)
* Camera module

All components are integrated through an Arduino-based control system to manage sensor input and actuator output.

---

## Current Limitations

* Real-time object detection is **not yet implemented**.
* RAG system currently operates on **pre-recorded video** instead of live video streams.
* Prototype-scale deployment; not yet optimized for large commercial greenhouses.

---

## Future Improvements

* Integration of **real-time object detection** for plant disease and stress identification.
* Live camera feed processing with edge AI.
* Mobile or web-based dashboard for user interaction.
* Expansion to multiple plant zones and sensor networks.
* Integration of weather-based adaptive watering strategies.

## Technologies Used

* **Programming:** Python, Arduino, CUDA
* **Machine Learning:** Elastic Net, CNN, SqueezeNet
* **Hardware:** Arduino, sensors, motors
* **AI System:** Retrieval-Augmented Generation (RAG)
* **APIs:** External AI/vision APIs (2 used)

---

## Project Status

🚧 **Prototype Stage**
This project is a functional prototype that demonstrates the feasibility and core concepts. Further development is planned.

## Contributors

* **Islam** – Dataset Collection, System design, ML modeling, Hardware setup, Arduino integration, sensor calibration, Motor Setup
* **Maria** – Dataset Collection, AI integration

