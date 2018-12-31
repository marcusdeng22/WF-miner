# WF-miner

Radial line detection for mining in the game Warframe. This is not intended for cheating but as a fun application to develop.



## Goal

To automate and improve mining accuracy in Warframe using computer vision. Mining in Warframe is explained well [here](https://warframe.fandom.com/wiki/Mining). Holding the mouse down on a mining node will create a circular bar with up to two moving brackets on the progress bar that slowly fills up; releasing the mouse when the progress bar is in the center of a bracket (target) will yield a rarer gem or more minerals than if the mouse were released elsewhere.



## Usage

Run live-target-multi.py using Python3 and go mining!



## How it works

The Python script depends on [MSS](https://github.com/BoboTiG/python-mss) and [OpenCV](https://opencv.org/). The MSS library takes screenshots of the center of the gameplay screen and the resulting images are processed in OpenCV. Hough line detection is applied on the radial lines and lines passing through the center of the image are used in calculating the estimated center of the target. Comparisons are also made between the image and a template image to determine the current progress bar location. When the progress bar is close to the center of the target, a mouse up event is triggered to complete the mining.



## Limitations

This script is only configured for a monitor with resolution 1600x900. Additionally, depending on computational resources, the script will release the mouse too early or too late. Progress is being made to make each iteration more efficient and strict to improve timing.