import pathlib
from qutils import save_checkpoint, load_checkpoint, get_diffusion_batch
import torch
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig

from torch.utils.data import DataLoader
import torchvision
import einops
import sys
import numpy as np
import pygame
import time


import pygame
from time import sleep
import threading
from pygame.locals import * 
import numpy as np
import time

class JoyManager():

    def __init__(self, user_callback = None):
        self.dt = 0.01 #Sampling frequence of the joystick
        self.runnign = False
        self.joy_thread_handle = threading.Thread(target=self.joy_thread)
        #an optional callback that can be used to send the reading values to a user provided function
        self.user_callback = user_callback
        pygame.init()
        self.offset = None

    def joy_thread(self):
        while self.running:   
            for event in [pygame.event.wait(200), ] + pygame.event.get():
                    # QUIT             none
                    # ACTIVEEVENT      gain, state
                    # KEYDOWN          unicode, key, mod
                    # KEYUP            key, mod
                    # MOUSEMOTION      pos, rel, buttons
                    # MOUSEBUTTONUP    pos, button
                    # MOUSEBUTTONDOWN  pos, button
                    # JOYAXISMOTION    joy, axis, value
                    # JOYBALLMOTION    joy, ball, rel
                    # JOYHATMOTION     joy, hat, value
                    # JOYBUTTONUP      joy, button
                    # JOYBUTTONDOWN    joy, button
                    # VIDEORESIZE      size, w, h
                    # VIDEOEXPOSE      none
                    # USEREVENT        code
                    if event.type == QUIT:
                        self.stop_daq()
                    elif event.type == KEYDOWN and event.key in [K_ESCAPE, K_q]:
                        self.stop_daq()
                    elif event.type == JOYAXISMOTION:
                        self.analog_cmd[event.axis] = event.value
                    elif event.type == JOYBUTTONUP:
                        self.digital_cmd[event.button] = 0
                    elif event.type == JOYBUTTONDOWN:
                        self.digital_cmd[event.button] = 1
                    if self.user_callback is not None and event.type!=NOEVENT:
                        if self.offset is None:
                            self.user_callback(self.analog_cmd, self.digital_cmd)
                        else:
                            self.user_callback(np.array(self.analog_cmd)-self.offset, self.digital_cmd)

                
    def start_daq(self, joy_idx):
        #Get the joy object
        assert pygame.joystick.get_count() != 0, 'No joysticks detected, you can not start the class'
        assert pygame.joystick.get_count() >= joy_idx, 'The requested joystick ID exceeds the number of availble devices'
        self.joy = pygame.joystick.Joystick(joy_idx)
        
        self.analog_cmd = [self.joy.get_axis(i) for i in range(self.joy.get_numaxes())]
        self.digital_cmd = [self.joy.get_button(i) for i in range(self.joy.get_numbuttons())]
        self.running = True
        self.joy_thread_handle.start()
        
    def stop_daq(self):
        self.running = False
        self.joy_thread_handle.join()
        
    def read_raw(self):
        return self.analog_cmd, self.digital_cmd

    def offset_calibration(self):
        analog , _ = self.read_raw()
        offset = np.array(analog)
        print('Put your stick at reset and do not touch it while calibrating')
        sleep(1)
        for i in range(2000):
            sleep(0.001)
            analog , _ = self.read_raw()
            offset += np.array(analog)
        self.offset = offset / 2000

    def read_values(self):
        analog, digital = self.read_raw()
        if self.offset is None:
            return analog, digital
        else:
            return analog-self.offset, digital
        
class XBoxController(JoyManager):
    def __init__(self, joystick_index):
        super(XBoxController, self).__init__()
        self.start_daq(joystick_index)
        time.sleep(0.5)
        try:
            self.offset_calibration()
        except:
            raise Exception(f'Could not communicate with the joystick with index: {joystick_index}')
        
    def getStates(self, deadzone=0.1):
        analog, digital = self.read_values()
        for i in [0,1,3,4]:
            if np.abs(analog[i])<=deadzone:
                analog[i]=0

        left_joy = analog[0:2]
        right_joy = analog[3:5]
        left_joy[1] =  -left_joy[1]
        right_joy[1] = -right_joy[1]
        left_trigger = analog[2]
        right_trigger = analog[5]
        A, B, X, Y = digital[0:4]
        left_bumper, right_bumper = digital[4:6]
        options_left, options_right = digital[6:8]
        left_joy_btn, right_joy_btn = digital[9:]

        return {
            'left_joy':left_joy,
            'right_joy':right_joy,
            'left_trigger':left_trigger,
            'right_trigger':right_trigger,
            'A':A,
            'B':B,
            'X':X,
            'Y':Y,
            'left_bumper':left_bumper,
            'right_bumper':right_bumper,
            'options_left':options_left,
            'options_right':options_right,
            'left_joy_btn':left_joy_btn,
            'right_joy_btn':right_joy_btn
        }
    def close(self):
        self.stop_daq()
            




controller = XBoxController(0)


state = controller.getStates()
v = state['right_joy']






sys.path.append("/home/quim/code/diffusion_planning_v2/src")
import qdataset

# Load the world model.
path = "qoutput/2024-12-03/11-04-29_l75jja/ckpt/ckpt_00000027.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = load_checkpoint(path, device)
model = D["model"]
diffusion_sampler = DiffusionSampler(denoiser=model, cfg=D["diffusion_sampler_cfg"])

datapath = "/home/quim/code/diffusion_planning_v2/data/pusht_more_data/traj_v3/"

tgt_traj_len = 16
dataset = qdataset.DiskDatasetTrajv2(datapath, tgt_traj_len=tgt_traj_len)

batch_size = 32  # Since we're creating a game, we only need one instance
dataloader = DataLoader(
    dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True
)

# Get initial batch
batch = next(iter(dataloader))

batch_diffusion = get_diffusion_batch(
    batch, act_is_vel=True, act_is_vel_discretized=False, device=device
)

num_steps_conditioning = D["denoiser_cfg"].inner_model.num_steps_conditioning

prev_obs = batch_diffusion.obs[:, :num_steps_conditioning]
prev_act = batch_diffusion.act[:, :num_steps_conditioning]

# Initialize action
action = torch.zeros(batch_size, 2).to(device)

# Initialize pygame
pygame.init()

# Set up display
# Assuming images are square and dimensions match your model's output
img_height = prev_obs.shape[-2]
img_width = prev_obs.shape[-1]

# Set the scaling factor
scale_factor = 5

# Calculate the new window size
window_width = img_width * scale_factor
window_height = img_height * scale_factor
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('World Model Game')

clock = pygame.time.Clock()
FPS = 10  # Run at 10 Hz

running = True

model = torch.compile(model) 
action = torch.zeros(batch_size, 2).to(device)


while running:
    clock.tick(FPS)  # Limit to FPS
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Process keyboard input
    keys = pygame.key.get_pressed()
    
    speed = 0.05  # Adjust the speed scaling as needed
    
    if keys[pygame.K_LEFT]:
        action[0, 0] -= speed  # Decrease x velocity
    if keys[pygame.K_RIGHT]:
        action[0, 0] += speed  # Increase x velocity
    if keys[pygame.K_UP]:
        action[0, 1] -= speed  # Decrease y velocity
    if keys[pygame.K_DOWN]:
        action[0, 1] += speed  # Increase y velocity
    
    # Clip action to valid range if necessary
    action = torch.clamp(action, min=-1.0, max=1.0)
    
    # Sample next observation
    tic = time.time()
    with torch.no_grad():
        out = diffusion_sampler.sample(prev_obs=prev_obs, prev_act=prev_act)
    toc = time.time()
    # print("Time taken to sample:", toc - tic)
    # print((toc-tic)/batch_size)
    new_obs = out[0]
    
    # Convert new_obs to image
    img_tensor = 0.5 * (new_obs[0] + 1)  # Scale from [-1,1] to [0,1]
    img_array = img_tensor.detach().cpu().numpy()
    img_array = np.transpose(img_array, (1, 2, 0))  # (H, W, C)
    img_array = (img_array * 255).astype(np.uint8)
    if img_array.shape[2] == 1:
        img_array = np.repeat(img_array, 3, axis=2)  # Convert grayscale to RGB
    
    # Create a pygame Surface
    surface = pygame.surfarray.make_surface(img_array)
    surface = pygame.transform.rotate(surface, -90)  # Adjust if image is rotated
    surface = pygame.transform.flip(surface, True, False)  # Adjust if image is flipped
    
    # Scale the surface
    scaled_surface = pygame.transform.scale(surface, (window_width, window_height))
    
    # Blit to screen
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()
    
    # Update prev_obs and prev_act
    prev_obs = torch.roll(prev_obs, shifts=-1, dims=1)
    prev_obs[:, -1] = new_obs
    prev_act = torch.roll(prev_act, shifts=-1, dims=1)


    state = controller.getStates()
    v = torch.tensor(state['right_joy'])
   
    v_norm =  .5  * v 
    v_norm[1] = -v_norm[1]

    action = v_norm

    print("action", action)




    prev_act[:, -1] = action

pygame.quit()
