# GestureCAD

## AI-Powered Hand Gesture Controlled 2D & 3D Graphics Editor

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![NumPy](https://img.shields.io/badge/NumPy-Mathematics-yellow)
![License](https://img.shields.io/badge/License-MIT-purple)

GestureCAD is a real-time hand gesture-controlled graphics editor that enables users to create, manipulate, and transform 2D drawings into interactive 3D objects using natural hand movements captured through a standard webcam.

Developed using Python, OpenCV, MediaPipe, and NumPy, GestureCAD demonstrates the integration of Computer Vision, Human-Computer Interaction (HCI), and Real-Time Graphics Rendering to provide an intuitive CAD-like drawing experience.

One of the key features of GestureCAD is its automatic extrusion system, where a freehand 2D drawing is converted into a 3D model when the user rotates the selected object.

---

# Features

## Drawing Tools

- Freehand Drawing
- Rectangle
- Triangle
- Circle

## Built-in 3D Objects

- Cube (Box)
- Sphere
- Cylinder
- Triangular Prism

## Object Manipulation

- Draw objects using hand gestures
- Select and move objects
- Resize geometric shapes
- Rotate objects in three-dimensional space
- Automatic 2D to 3D extrusion

## Interaction Modes

- Hand Gesture Control
- Mouse Control

---

# Technologies Used

- Python 3.11+
- OpenCV
- MediaPipe Hands
- NumPy

---

# Project Structure

```text
GestureCAD/
│
├── gesture_cad.py
├── requirements.txt
├── models/
│   └── hand_landmarker.task
│
├── assets/
│   ├── demo.gif
│   ├── screenshots/
│   └── icons/
│
├── README.md
└── LICENSE
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/GestureCAD.git
```

Navigate to the project directory:

```bash
cd GestureCAD
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python gesture_cad.py
```

---

# First Launch

During the first execution, GestureCAD automatically downloads the official MediaPipe Hand Landmark model (approximately 8 MB) into the local `models` directory.

This download occurs only once.

No additional files are downloaded by the application.

---

# Using a Different Camera

If your webcam is not detected as Camera 0, specify the camera index manually.

```bash
python gesture_cad.py --camera 1
```

---

# Self-Test

GestureCAD includes a built-in self-test mode to verify:

- 2D Extrusion
- Cube Mesh Generation
- Sphere Mesh Generation
- Cylinder Mesh Generation
- Triangular Prism Mesh Generation

Run the self-test:

```bash
python gesture_cad.py --self-test
```

---

# Hand Gestures

| Gesture | Function |
|----------|----------|
| Index Finger | Freehand Drawing |
| Thumb + Index Pinch | Select Toolbar Tool |
| Pinch + Drag | Resize Shapes |
| Pinch in Select Mode | Select and Move Objects |
| Index + Middle Fingers | Rotate Selected Object in 3D |

---

# Mouse Controls

| Action | Function |
|--------|----------|
| Left Click | Select Tool |
| Left Click + Drag | Draw or Move Objects |
| Right Click + Drag | Rotate Selected Object |
| Q or Escape | Exit Application |

---

# Workflow

1. Draw a freehand object or geometric shape.
2. Select the object.
3. Rotate the object using the designated gesture.
4. GestureCAD automatically extrudes the 2D drawing into a 3D object.
5. Continue rotating and manipulating the generated model in real time.

---

# System Architecture

```text
Webcam Input
      │
      ▼
MediaPipe Hand Detection
      │
      ▼
Gesture Recognition
      │
      ▼
Shape Generation
      │
      ▼
Object Manipulation
      │
      ▼
2D to 3D Extrusion
      │
      ▼
Perspective Projection
      │
      ▼
Real-Time Rendering
```

---

# Learning Outcomes

This project provided practical experience in:

- Computer Vision
- Human-Computer Interaction (HCI)
- Gesture Recognition
- OpenCV Programming
- MediaPipe Hands
- Real-Time Graphics Rendering
- 3D Geometry
- Mesh Generation
- Perspective Projection
- Performance Optimization
- Interactive User Interface Design

---

# Future Enhancements

- Multi-hand support
- Undo and Redo functionality
- Save and Load Projects
- Export models as OBJ/STL
- Text annotation tools
- Lighting and Shadow Effects
- Texture Mapping
- OpenGL-based Rendering
- Voice Commands
- Multi-object Selection
- Gesture Customization

---

# Performance

- Real-time webcam processing
- Lightweight architecture
- Optimized rendering pipeline
- Smooth gesture recognition
- Compatible with standard CPU-based systems

---

# Requirements

- Python 3.11 or later
- Webcam
- Windows, Linux, or macOS
- Internet connection (required only for the initial MediaPipe model download)

---

# Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.

```bash
git checkout -b feature/NewFeature
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push your branch.

```bash
git push origin feature/NewFeature
```

5. Open a Pull Request.

---

# Acknowledgements

This project makes use of the following open-source technologies:

- OpenCV
- MediaPipe
- NumPy
- Python

Special thanks to the open-source community for providing these excellent tools and libraries.

---

# License

This project is licensed under the MIT License.

---

# Developer

**Nausheen Fathima A**

**B.Tech Artificial Intelligence and Data Science**

### Areas of Interest

- Artificial Intelligence
- Computer Vision
- Human-Computer Interaction
- Machine Learning
- Deep Learning
- Graphics Programming
- Real-Time Interactive Systems

---

# Output
<img width="1787" height="831" alt="image" src="https://github.com/user-attachments/assets/0fc56d1e-4a76-4bb7-be36-879a32e1f7ff" />
