# laser-scanning

A 3D Scanner using Laser Structured Light, written in Python using OpenCV and NumPy.

### Input:

![input](./docs/input.gif)

### Output (in PLY format):

![output](./docs/output.gif)

*Source footage thanks to professor Filippo Bergamasco (Ca Foscari University of Venice).*

## Usage

### Camera Calibration

Before capturing the 3D model, a Camera Intrinsics calibration is expected. For this task you need to have some images with a calibration chessboard pattern. The calibration pattern is available in `patterns/chessboard.pdf`.

![calibration pattern](./docs/example_pattern.png)

After capturing the images, put them in the `calibration_images` folder and run this command:

```
python3 cameraCalibrator.py -v
```

You can omit the `-v` option if you don't want to see debug output.

After the calibration is completed the result will be saved in `intrinsics.xml`, ready to be loaded by the scanner.

### Scanner

To start the scanner with a video file, run this command:

```
python3 scanner.py -v cup1.mp4
```

You can omit the `-v` option if you don't want to see debug output.

The resulting scan will be showed after finish and will be saved in `output.ply`.

You may need to adjust the HSV ranges for the laser in the head of `scanner.py`.
