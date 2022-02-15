#!/home/bfd/venvs/raft/bin/python

""" Node Commands:
RaftBridge: start
RaftBridge: stop
"""

import sys
sys.path.append('core')

import os
import re
import cv2
import yaml
import torch
import rospy
import numpy as np
import time
from typing import NamedTuple

from cv_bridge import CvBridge, CvBridgeError

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from argparse import Namespace
from sensor_msgs.msg import Image
from std_msgs.msg import String


class RaftBridge:
    def __init__(self, config):
        rospy.init_node('raft_bridge', anonymous=False)

        self.config = config
        self.latest_update = 0
        self._isActive = False
        self._initRaft(config['raft'])
        self._initBridge({**config['topics'], **config['general']})


    def _initRaft(self, config):
        # Load Model etc
        args = Namespace(model = config['model'],
                         small=config['small'],
                         alternate_corr = False,
                         mixed_precision = False)

        self.device = config['device']
        self.width  = config['width']
        self.height = config['height']
        self.flow_width  = config['image_width']//2
        self.flow_height = config['image_height']//2
        self.image2 = np.zeros((self.flow_height, self.flow_width, 3), dtype=np.uint8)
        self.image2 = torch.from_numpy(self.image2).permute(2, 0, 1).float()
        self.image2 = self.image2[None].to(self.device)

        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))

        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

        self.padder = InputPadder([self.flow_width, self.flow_height])


    def _initBridge(self, config):
        # Where to subscribe/publish
        self.master_sub = rospy.Subscriber("%s" % config['master'],
                                            String,
                                            self.callbackMaster,
                                            queue_size=10,
                                            tcp_nodelay=True)

        self.readback_pub = rospy.Publisher("%s" % config['readback'],
                                           String,
                                           queue_size=1,
                                           tcp_nodelay=True)

        self.image_sub = rospy.Subscriber("/%s/%s" % (config['quad_name'],
                                                      config['camera_images']),
                                           Image,
                                           self.callbackImage,
                                           queue_size=1,
                                           tcp_nodelay=True)

        self.flow_pub = rospy.Publisher("/%s/%s" % (config['quad_name'],
                                                       config['optical_flow']),
                                           Image,
                                           queue_size=1,
                                           tcp_nodelay=True)
        self.sim = config['sim']
        self.bridge = CvBridge()


    def computeFlow(self, image1, image2):
        # Call RAFT model
        with torch.no_grad():
                image1, image2 = self.padder.pad(image1, image2)
                _, optical_flow = self.model(image1, image2, iters=20, test_mode=True)

        self.flow = optical_flow.squeeze().permute(1,2,0).cpu().numpy()


    def showFlow(self, image1, image2, optical_flow):
        # Visualize the result: two images and flow
        img1 = image1.squeeze().permute(1,2,0).cpu().numpy()
        img2 = image2.squeeze().permute(1,2,0).cpu().numpy()

        # map flow to rgb image
        flow = flow_viz.flow_to_image(self.flow)
        img_viz = np.concatenate([img1, flow], axis=0)

        cv2.imshow('image', img_viz[:, :, [2,1,0]]/255.0)
        cv2.waitKey(1)


    def callbackImage(self, data):
        if not self._isActive:
            return
        
        self.latest_update = data.header.stamp
        self.image1 = self.image2

        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        if self.sim:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img[:,:,0] = img_gray
            img[:,:,1] = img_gray
            img[:,:,2] = img_gray

        img = cv2.resize(img, (self.flow_width, self.flow_height))

        # write into image2
        img = np.array(img.data, ndmin=3).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        self.image2 = img[None].to(self.device)

        self.computeFlow(self.image1, self.image2)
        #  self.showFlow(self.image1, self.image2, self.flow)
        self.publish()


    def callbackMaster(self, msg_string):
        instruction = msg_string.data
        if not re.search(r"RaftBridge", instruction):
            return

        print("Raft Bridge received command '%s'" % instruction)
        if instruction == "RaftBridge: start":
            self._isActive = True
            self.readback_pub.publish(String(instruction))
        elif instruction == "RaftBridge: stop":
            self._isActive = False
            self.readback_pub.publish(String(instruction))
        else:
            print("Command not understood.")


    def publish(self):
        #  flow = np.zeros((self.height, self.width, 2), dtype=self.flow.dtype)
        flow = cv2.resize(self.flow, (self.width, self.height))
        u = flow[:,:,0]
        v = flow[:,:,1]
        mag = np.sqrt(np.square(u) + np.square(v))
        angle = np.cos(np.arctan2(v,u))

        flow_out = np.stack((mag, angle), axis=-1)
        image_message = self.bridge.cv2_to_imgmsg(flow_out, encoding="passthrough")
        image_message.header.stamp = self.latest_update
        self.flow_pub.publish(image_message)
        


if __name__=="__main__":
    with open('../settings.yaml') as f:
        config = yaml.safe_load(f)

    RaftBridge(config)
    rospy.spin()
