# Movehack
# Giving Senses
## We propose smart analytics for different senses such as vision, hearing,etc to derive productive data from those, which can be further linked with Data Science (ML, DL, AI). The reason behind using various senses is to increase the accuracy and efficiency for a specific application. The technology behind video analytics is Yolov (i.e 1000x faster than current models being used i.e R-CNN). By this we will be able to generate data sets with very less cost for different senses. It can be used in various places such Autonomous Vehicle  , traffic flow mapping,etc

## Dependencies

Python3, tensorflow 1.0, numpy, opencv 3.

## Installing

1. Just build the Cython extensions in place. 
    ```
    python3 setup.py build_ext --inplace
    ```

2. Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip install -e .
    ```

3. Install with pip globally
    ```
    pip install .
    ```

##Running
1. Just type
    ```
    python3 Webcam.py
    ```
    
## Output
1.  Results will start displaying on the terminal.

2. Data in json is stored in kishan.txt in DATA folder.

3. Images can be seen in images-by-timeframe folder



