# Blink-comparator
A blink comparator is a device used in astronomy to compare two images of the night sky taken at different times [(Wikipedia page)](https://en.wikipedia.org/wiki/Blink_comparator). It consists of two separate viewing scopes that alternately display the two images by rapidly blinking back and forth between them. Astronomers use this tool to detect changes, such as the movement of celestial objects like asteroids or variable stars, by observing any differences or shifts between the images. This allows for the identification of objects that have changed position or brightness over time.

## Description of the code
This Python script implements a blink comparator using pure computer vision algorithms.

The two images passed as input will be compared and the similarities highlighted. Furthermore, the two images will be superimposed and alternated suddenly so as to make the differences visible to the human eye.

In addition, the code also calculates the number of celestial bodies present in the images, so as to provide further data to the user.

## Installation
```bash
git clone https://github.com/federicomigliosi/Blink-comparator
cd Blink-comparator/
pip install -r requirements.txt
```
## Usage
In order to run the script, execute the following command:
```bash
python3 blink_comparator.py [LeftImage] [RightImage]
```
The two input parameters ```LeftImage``` and ```RightImage``` correspond to the two different astronomical images of the night sky taken at different times.

## Credits
This script was created with the help of the following [guide](https://nostarch.com/real-world-python).
