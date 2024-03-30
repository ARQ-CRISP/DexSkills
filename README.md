# DexSkill
[WORK IN PROGRESS]
Effective execution of long-horizon tasks with dexterous robotic hands remains a significant challenge in real-world problems. While learning from human demonstrations have shown encouraging results, they require extensive data collection for training. Hence, decomposing long-horizon tasks into reusable primitive skills is a more efficient approach. To achieve so, we developed DexSkills, a novel supervised learning framework that addresses long-horizon dexterous manipulation tasks using primitive skills. DexSkills is trained to recognize and replicate a select set of skills using human demonstration data, which can then segment a demonstrated long-horizon dexterous manipulation task into a sequence of primitive skills to achieve one-shot execution by the robot directly. Significantly, DexSkills operates solely on proprioceptive and tactile data, i.e., haptic data. Our real-world robotic experiments show that DexSkills can accurately segment skills, thereby enabling autonomous robot execution of a diverse range of tasks.

Video Submitted

### Intro
![First_image_lightC](https://github.com/ARQ-CRISP/DexSkill/assets/62802841/fe441aaa-b638-4bbb-aa16-a87db6b6d2b3)


The dataset includes data of 20 haptic skils (10 repetitions each):
1.  Reach
2.  Setup Position
3.  Pretouch
4.  Touch
5.  Flip
6.  Wipe Forth
7.  Wipe Back
8.  PreGrasp
9.  Grasp
10. Lift with Grasp
11. Transport Forward
12. Place
13. PreRotate
14. Rotate
15. Shake Up
16. Shake Down
17. Twist
18. Vertical Place
19. Pour
20. Release

And N Long Tasks executed as a sequence of skills.


### Method

### Dataset Links

### Data Modalities
The dataset provides the following modalities:

 - Proprioception
 - Tactile Sensing

For every every skill the date files are organised as following:

NB Edit according to your dataset disposition.
```
avocado
    └─ single
         └── 1
             ├── allegro_fingertips.csv
             ├── allegro_joints.csv
             ├── camera_compressed
             │   ├── 1634755101828721375.jpg
             │   ├── 1634755101861649368.jpg
             │   ├── 1634755101893623525.jpg
             │   ├── 1634755101927505682.jpg
             │   ├── .................
             │   ├──
             │
             ├── labels
             ├── leap_fingertips.csv
             ├── optoforce_data.csv
             └── ur5_joints.csv
```

### Data Files


The data files are expressed as following: 
``` 
Express every column of the data file and the format
```

### Demo Videos

https://github.com/ARQ-CRISP/DexSkill/assets/62802841/f60f5a8a-f3ac-4726-8840-50d545b14b38

https://github.com/ARQ-CRISP/DexSkills/assets/62802841/f7fc40b3-19dd-41a2-95a5-e7b9d42a9b8d

