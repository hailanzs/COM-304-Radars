from multiprocessing import Process, Queue
import logging
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
import matplotlib.pyplot as plt
import numpy as np
import realtime_streaming as rtstream
import radar as radar
import os
from multiprocessing import Process, Queue
import logging
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
import matplotlib.pyplot as plt
import numpy as np
from streaming.prod_dca import producer_real_time_1843


def consumer(q, index):
    app = MyApp(q)
    app.run()

def main(exp_num, radar1):

    num_producers = 1
    num_consumers = 1
    max_queue_size = 10000

    logging.info(r"Execution happening with %d producers and %d consumers" % (num_producers, num_consumers))

    q_main = Queue(maxsize=max_queue_size)

    producers = []
    consumers = []
    producers.append(Process(target=producer_real_time_1843, args=(q_main, 0, radar1)))

    
    # Create consumer processes
    for i in range(0, num_consumers):
        p = Process(target=consumer, args=(q_main, 0))
        p.daemon = False

        consumers.append(p)

    # Start the producers and consumer
    # The Python VM will launch new independent processes for each Process object
    for p in producers:
        p.start()
 
    for c in consumers:
        c.start()
 
    # Like threading, we have a join() method that synchronizes our program
    try:
        for p in producers:
            p.join()

        # Wait for empty chunks queue to exit
        for c in consumers:
            c.join()
    except KeyboardInterrupt as ki:
        print(r"Program terminated by keyboard")

    logging.info('Parent process exiting...')
    # update status to done, or cancelled or error?

class MyApp(ShowBase):
    def __init__(self, queue):
        ShowBase.__init__(self)

        self.q = queue
        # self.accept("escape", sys.exit, [0])

        #NEED TO CHANGE THESE EACH TIME
        self.rfft_size = 512
        self.rfft_range = [0, 1]

        # Create subplots for FFT and phase 
        plt.ion()
        self.fig = plt.figure()
        self.ax_rfft = self.fig.add_subplot(111)
        self.text_rfft = self.ax_rfft.text(0.90, 0.85, "Nothing" , fontsize=40, transform=self.ax_rfft.transAxes, verticalalignment='top', ha='right')

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed') #works fine on Windows!

        # initialize FFT plot
        self.rfft_x_data = np.arange(self.rfft_size)
        self.rfft_y_data = np.zeros_like(self.rfft_x_data)
        self.line_rfft = self.ax_rfft.stem(self.rfft_x_data, self.rfft_y_data)
        self.ax_rfft.set_ylim(self.rfft_range)

        self.taskMgr.add(self.updateDataTask, "updateDataTask")
        self.taskMgr.add(self.updateRFFTPlotTask, "updateFFTPlotTask")

    def updateRFFTPlotTask(self, task):
        self.line_rfft[0].set_ydata(self.rfft_y_data)
        self.line_rfft[1].set_paths([np.array([[xx, 0], 
                                    [xx, yy]]) for (xx, yy) in zip(self.rfft_x_data, self.rfft_y_data)])
        self.ax_rfft.set_ylim([np.min(self.rfft_y_data)-1, np.max(self.rfft_y_data)+1])
        max_ind = np.argmax(self.rfft_y_data)
        self.text_rfft.remove()
        self.text_rfft = self.ax_rfft.text(0.56, 0.85, "Here is some text", transform=self.ax_rfft.transAxes, fontsize=15,verticalalignment='top', ha='left')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return Task.cont

    def updateDataTask(self, task):
        try:
            while self.q.qsize() > 0:
                self.counter += 1
                new_data = self.q.get(block=False)
                if new_data[0] == "rfft":
                    self.rfft_y_data = new_data[1]
                
        except:
            return Task.cont

        return Task.cont


## TODO: Code to edit starts here!!!
rtt_path = r'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll'
# create folders for saving data
home_dir = r'C:\Users\robin\Documents\EPFL\BA5\Comm_project\comm-proj-radar' # home directory path (of the project folder, full path)
exp_path = os.path.join(home_dir, r'record') # put the folder you want to save data to (full path)
exp_name = r'test' # name of the file you want to save (no .bin) in this case this is a dummy file that you will dump data bc TI makes us
config_lua_script = r'scripts/1843_config_streaming.lua' # relative path to the lua scripts for continuous it in is the home dir(ex. scripts/1843_config_streaming.lua)
record_lua_script =r'scripts/1843_record.lua' # relative path to the lua scripts for recording it in is the home dir(ex. scripts/1843_record.lua)

# initialize the radar class, this runs .lua scripts from mmWaveStudio and captures data
radar1 = radar.radar(home_dir, rtt_path,config_lua_script)

# check if you are maybbe accidently overwriting a file
if os.path.isfile((os.path.join(exp_path, r"%s.bin" % exp_name))):
    print("You have files created already so you might overwrite data!")
else:
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)

    # capture data and process it
    radar1.mmwave_capture(exp_name, exp_path,record_lua_script)

# To edit the plots and how they show up go to streaming._realtime_streaming.py
# Here we will run the plotting code to see data in realtime
rtstream.stream(1, radar1)




if __name__ == '__main__':
    main(0,radar1)
