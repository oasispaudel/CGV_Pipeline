CGV Pipeline Explanation (Step-by-Step)

This document explains the CGV pipeline stages:
Original -> Model -> World -> World -> View -> Projection (NDC) -> Clipped (NDC) -> Viewport (Screen)

It is written for beginners and includes real-world analogies and current usage.

1) Original (Model Space)
What it means:
The object in its own local coordinates, as it was created. Example: a unit cube centered at the origin.

Example:
A cube is defined with vertices from (-0.5, -0.5, -0.5) to (0.5, 0.5, 0.5). This is a clean, simple coordinate system that makes modeling easy.

Real-world analogy:
Imagine a toy cube on a table. The measurements you use to build the toy in a workshop are its model space.

Why it matters:
Artists and engineers design objects in their own local space because it is simpler and reusable. The same model can be placed in many scenes.

2) Model -> World (World Space)
What it means:
We move, rotate, and scale the object so it sits correctly inside the scene.

Example:
Take the unit cube and move it 2 units to the right, rotate it 30 degrees, and scale it up by 1.5. Now it is placed in the world.

Real-world analogy:
You place the toy cube on a shelf, turn it slightly, and maybe choose a bigger version.

Why it matters:
World space is the shared coordinate system where all objects are positioned together (buildings, cars, characters, etc.).

3) World -> View (Camera Space)
What it means:
We convert the world so that the camera becomes the origin (0,0,0) and looks down the -Z axis.

Example:
If the camera is at (3, 3, 4) looking at the origin, the world is transformed so the camera is "at the center," and everything else moves accordingly.

Real-world analogy:
You move your head and look at the toy. The toy is still in the same place, but your viewpoint changes.

Why it matters:
Rendering is easiest when everything is expressed relative to the camera.

4) Projection (NDC)
What it means:
We map 3D camera-space coordinates into Normalized Device Coordinates (NDC), which is a standard cube:
X, Y, Z in the range [-1, 1].

Example:
After projection, a vertex might become (0.8, -0.2, 0.4). This means it is inside the visible cube.

Real-world analogy:
Think of a camera lens that squeezes the 3D world into a flat image while keeping a standard "measurement box."

Why it matters:
All screens and devices can use the same normalized space before converting to pixels.

5) Clipped (NDC)
What it means:
Any part of the object outside the visible NDC cube is cut away. Only what can be seen remains.

Example:
If a line crosses outside the NDC bounds, it is trimmed so only the visible part stays.

Real-world analogy:
It is like looking through a window. Anything outside the window frame is hidden.

Why it matters:
Clipping avoids drawing invisible geometry, making rendering faster and correct.

6) Viewport (Screen)
What it means:
We map NDC into actual screen pixel coordinates based on the viewport size.

Example:
NDC (0,0) might become screen pixel (400, 300) if the viewport is 800x600.

Real-world analogy:
We print the final image onto a specific screen resolution or window size.

Why it matters:
This step finally connects the math pipeline to real pixels that appear on the display.

How this pipeline is used today (Current Scenario)
- Video games: The pipeline runs every frame for thousands of objects.
- CAD/Engineering: Accurate orthographic projection is used for technical drawings.
- Simulation/Robotics: Camera models and projection help interpret sensor views.
- VR/AR: View and projection stages are critical for stereo rendering and depth.
- Data visualization: 3D charts and graphs are projected and mapped to screens.

Quick Recap
- Model Space: object's own coordinates
- World Space: object placed in the scene
- View Space: scene relative to the camera
- NDC: normalized cube [-1,1] for visibility
- Clipping: remove invisible parts
- Viewport: map to screen pixels

If you want to link these steps to a screenshot:
- Original = Model Space
- Model -> World = World Space
- World -> View = Camera Space
- Projection (NDC) = NDC Space
- Clipped (NDC) = Clipped NDC
- Viewport (Screen) = Pixel coordinates
