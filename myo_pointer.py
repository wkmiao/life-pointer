
from __future__ import print_function
import myo as libmyo; libmyo.init("/Users/andywang/htn-pointy-thing/sdk/myo.framework/myo")


class Listener(libmyo.DeviceListener):

  def on_connected(self, event):
    print("Hello, '{}'! Double tap to exit.".format(event.device_name))
    event.device.vibrate(libmyo.VibrationType.short)
    event.device.request_battery_level()

  def on_battery_level(self, event):
    print("Your battery level is:", event.battery_level)

  def on_pose(self, event):
    if event.pose == libmyo.Pose.wave_in:
      print ("left")
      return True
    elif event.pose == libmyo.Pose.wave_out:
      print ("right")
      return True
  
    elif event.pose == libmyo.Pose.double_tap:
      return False



if __name__ == '__main__':
  hub = libmyo.Hub()
  listener = Listener()
  while hub.run(listener.on_event, 500):
    pass
  print('Bye, bye!')