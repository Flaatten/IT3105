"""
possible input should be:
a.  hl = Nr of hidden layers
b.  haf = Hidden activation function
c.  oa = Output activation
d.  cf = Cost function
e.  lr = Learning rate
f.  Initial weight range
g.  Data source – one of:
    i.  File 
    ii. Function (see tflowtools.py)
h.  Case fraction
i.  Validation fraction (VaF = [0,1])
    i.  S*VaF
j.  Test fraction (TeF = [0,1])
    i.  S*TeF
k.  Training fraction (1-(VaF+TeF))
    i.  S*(1 –(TeF+VaF))
l.  Minibatch size
m.  Map batch size
n.  Steps
o.  Map Layers
p.  Map dendrograms
q.  Display weights
r.  Display biases
s.  Vint(validation interval)
t.  UNSURE about these: 
    i.  Nodes in each layer
    ii. Error function
    iii.  Initial weight values
"""
import sys
sys.path.append('/Users/jonasdammen/Projects/IT3105')
sys.path.append('/Users/jonasdammen/Projects/IT3105/tools')
import tkinter as tk
from gann import Gann, Caseman
import tflowTools as TFT

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()
        self.arguments = [["Nr of hidden layers", "0"], ["Learning rate", "0.5"]]

    def create_widgets(self):
        """
        self.labels=[]
        for i in range(0,25):
            label = tk.Label(self)
            label.pack(side="left")
            self.labels.append()
        """
        self.entry = tk.Entry(self, )
        self.entry.pack(side="bottom")

        self.run_ANN = tk.Button(self)
        self.run_ANN["text"] = "Run ANN"
        self.run_ANN["command"] = self.run_network
        self.run_ANN.pack(side="left")

        self.init_ANN = tk.Button(self)
        self.init_ANN["text"] = "Initialize ANN"
        self.init_ANN["command"] = self.initialize_network
        self.init_ANN.pack(side="right")


        self.quit = tk.Button(self, text="QUIT", fg="red", command=root.destroy)
        self.quit.pack(side="bottom")

    def initialize_network(self, epochs=50,nbits=10,ncases=500,lrate=0.5,showint=500,mbs=20,vfrac=0.1,tfrac=0.1,vint=200,sm=True,bestk=1):
        print("initializing network with following input: ")
        print(self.entry.get())

        for i, argument in enumerate(self.arguments):
          print(str(i + 1) + ".  " + argument[0] + ": " + argument[1])
        case_generator = (lambda: TFT.gen_vector_count_cases(ncases,nbits))
        cman = Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
        self.ann = Gann(dims=[nbits, nbits*3, nbits+1], cman=cman, lrate=lrate, showint=showint, mbs=mbs, vint=vint, softmax=sm)
    

    def run_network(self,epochs=50, bestk=None):
        print("Running network with following input: ")
        for i, argument in enumerate(self.arguments):
          print(str(i + 1) + ".  " + argument[0] + ": " + argument[1])
        self.ann.run(epochs,bestk=bestk)

root = tk.Tk()
app = Application(master=root)
app.mainloop()

