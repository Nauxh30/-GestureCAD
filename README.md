# GestureCAD

GestureCAD is a real-time webcam graphics editor controlled by natural hand
gestures. Draw in 2D, then rotate the selected drawing to extrude it into 3D.
It also includes rectangle, triangle, circle, 3D box, triangular prism, sphere,
and cylinder tools.

## Install and run

Use Python 3.11 or newer:

```powershell
cd C:\Users\naush\GestureCAD
python -m pip install -r requirements.txt
python gesture_cad.py
```

The first run downloads MediaPipe's official hand-landmark model (about 8 MB)
to the local `models` folder. Nothing else is downloaded by the application.

If your webcam is not camera 0:

```powershell
python gesture_cad.py --camera 1
```

## Gestures

- **Index finger only:** draw with the Freehand tool.
- **Thumb + index pinch:** click a toolbar tool; pinch-drag to size a shape.
- **Pinch on an object in Select mode:** select and move it.
- **Index + middle fingers:** rotate the selected object in 3D. A 2D drawing
  is automatically extruded when rotation begins.

Every action also has a mouse fallback:

- Left click/drag to use tools, draw, create, select, or move.
- Right-drag to rotate the selected object.
- Press `Q` or `Escape` to quit.

Keyboard tool shortcuts are displayed in the toolbar.

## Test without a webcam

```powershell
python gesture_cad.py --self-test
```

This checks 2D extrusion plus box, prism, sphere, and cylinder mesh generation.
