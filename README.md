# Hack The North Project

Autonomy is important to one's self-empowerment and especially so if one already has other physical challenges. **LifePointer** is a tool for people with accessibility needs--specifically visual impairment--to detect objects around them in real time. With this hack, users can interact with the world using the speech and gesture-based user interface of this wearable technology, simply by pointing at objects around them to locate them.

Hardware setup:
![](https://raw.githubusercontent.com/MrMiaoMiao/life-pointer/master/demo/hardware-setup.jpg)

User-wearable setup:
![](https://raw.githubusercontent.com/MrMiaoMiao/life-pointer/master/demo/wearable-setup.jpg)

## How we built it
We used **Tensorflow's SSD Mobilenet** pretrained model for generic object recognition, combining it with **OpenCV** to detect the pointing finger. The device uses speech-based user interaction through the help of the **Google Speech API** for real-time speech-to-text transcription and result-to-speech output. Finally, user interface is heightened through the use of gesture controls, using the **Myo Arm Band**'s specific gestures to activate different features of this product.

Image detection results:
![](https://raw.githubusercontent.com/MrMiaoMiao/life-pointer/master/demo/image-detection-results.jpg)

## Challenges we ran into
Our biggest, and still ongoing challenge, is to strive for higher levels of accuracy and better performance. At first, we had difficulties to even detect a hand, much less the pointing direction of a finger. As well, there were a lot of hardware limitations if we were to follow our original plan of deploying the entire process independently onto a raspberry pi. Past the 20th hour, we ended up finally committing to using a laptop, with better processing abilities and better support for the Myo band, as well as more resources for the machine learning model.

## Accomplishments that we're proud of
Our main accomplishments consisted of improving the overall accuracy and performance of our hack: through different OpenCV transformations and fine-tuning, the precision of finer tracking was greatly increased. As well, our specific algorithm allows accurate detection of the pointed object, taking into account of the average of multiple frames, all while managing the different functionalities as a optimized work-around to multi-threading. All in all, when fully integrated together, it was optimized to reach a staggering 20 frames per second--reaching impressive levels of object detection and tracking of moving objects.


## What we learned
Integrating standalone units together takes at least ~~two~~ ~~five~~ a lot of times more time than expected.

## Some ambitious ideas for LifePointer
- Making the entire unit completely compact and wireless, allowing the users to freely move around and use this hardware in their day-to-day lives (the original idea was to make the whole process completely deployed on a raspberry pi, but due to processing and memory limitations, other workarounds had to be implemented)
- Stereo Video inputs for better intersection calculation between the pointing finger and the objects around the user,  to possibly triangulate the depth of the object and obtain higher accuracy
- Rely less on the green finger sleeve to be able to detect hand motion

## Notes to the developer: 
```
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/cred.json"
```

